"""Physical labels, fixed barrier features, and margin objectives.

Comfort labels describe measured temperature bounds, not a proved invariant set.
Boundary-adjacent samples are left unlabelled for certificate classification.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from datetime import date

import numpy as np
import torch
from torch import nn


@dataclass
class SafetySpec:
    temperature_variables: list[str]
    winter_bounds: list[float] = field(default_factory=lambda: [21.0, 24.0])
    summer_bounds: list[float] = field(default_factory=lambda: [23.0, 26.0])
    summer_start: list[int] = field(default_factory=lambda: [6, 1])
    summer_end: list[int] = field(default_factory=lambda: [9, 30])
    occupancy_variables: list[str] = field(default_factory=list)
    occupancy_threshold: float = 0.5
    unoccupied_bounds: list[float] = field(default_factory=lambda: [15.0, 30.0])
    label_guard_band_c: float = 0.2
    extra_features: dict = field(default_factory=dict)
    energy_variable: str = "HVAC_electricity_demand_rate"
    history_length: int = 1

    def __post_init__(self):
        if not self.temperature_variables:
            raise ValueError("Name at least one physical temperature variable")
        if len(set(self.temperature_variables)) != len(self.temperature_variables):
            raise ValueError("Temperature variables must be unique")
        if self.occupancy_variables and len(self.occupancy_variables) != len(self.temperature_variables):
            raise ValueError("Provide one occupancy variable per measured zone")
        for bounds in (self.winter_bounds, self.summer_bounds, self.unoccupied_bounds):
            if len(bounds) != 2 or not np.isfinite(bounds).all() or bounds[0] >= bounds[1]:
                raise ValueError(f"Invalid comfort bounds: {bounds}")
        if self.history_length < 1 or self.label_guard_band_c < 0:
            raise ValueError("Invalid history length or classification guard band")
        for name, pair in self.extra_features.items():
            if len(pair) != 2 or not np.isfinite(pair).all() or pair[1] <= 0:
                raise ValueError(f"Feature {name} requires [center, positive scale]")

    @property
    def frame_dim(self):
        return 4 * len(self.temperature_variables) + 4 + len(self.extra_features)

    @property
    def feature_dim(self):
        return self.frame_dim * self.history_length

    def required_variables(self):
        return (self.temperature_variables + self.occupancy_variables
                + list(self.extra_features) + [self.energy_variable, "month", "day_of_month", "hour"])

    def assess(self, raw: dict):
        missing = set(self.required_variables()) - set(raw)
        if missing:
            raise ValueError(f"Missing physical observations: {sorted(missing)}")
        if not all(np.isfinite(float(raw[k])) for k in self.required_variables()):
            raise FloatingPointError("Nonfinite physical safety observations")
        if float(raw[self.energy_variable]) < 0:
            raise ValueError("HVAC demand rate must be nonnegative power in watts")
        month, day = int(raw["month"]), int(raw["day_of_month"])
        md = (month, day)
        start, end = tuple(self.summer_start), tuple(self.summer_end)
        summer = start <= md <= end if start <= end else (md >= start or md <= end)
        seasonal = self.summer_bounds if summer else self.winter_bounds
        temps = np.array([raw[k] for k in self.temperature_variables], np.float32)
        occupied = (np.array([raw[k] for k in self.occupancy_variables]) > self.occupancy_threshold
                    if self.occupancy_variables else np.ones(len(temps), bool))
        bounds = np.array([seasonal if o else self.unoccupied_bounds for o in occupied], np.float32)
        low, high = bounds[:, 0], bounds[:, 1]
        signed = np.maximum(low - temps, temps - high)
        deviation = np.maximum(signed, 0)
        # ±epsilon losses cannot separate arbitrarily adjacent continuous labels.
        label = (1 if signed.max() > self.label_guard_band_c else
                 -1 if signed.max() < -self.label_guard_band_c else 0)
        midpoint, halfwidth = (low + high) / 2, (high - low) / 2
        features = []
        for i in range(len(temps)):
            features.extend([(temps[i] - midpoint[i]) / halfwidth[i],
                             (midpoint[i] - 22.0) / 10.0, halfwidth[i] / 10.0,
                             float(occupied[i])])
        hour = float(raw["hour"]) % 24
        doy = date(2000, month, day).timetuple().tm_yday
        features.extend([np.sin(2 * np.pi * hour / 24), np.cos(2 * np.pi * hour / 24),
                         np.sin(2 * np.pi * doy / 366), np.cos(2 * np.pi * doy / 366)])
        features.extend((float(raw[name]) - center) / scale
                        for name, (center, scale) in self.extra_features.items())
        return {"features": np.asarray(features, np.float32), "label": label,
                "deviation_c": deviation, "occupied": occupied, "bounds_c": bounds,
                "temperatures_c": temps, "power_w": float(raw[self.energy_variable])}


class BarrierNet(nn.Module):
    def __init__(self, input_dim, hidden_sizes=(64, 64)):
        super().__init__()
        layers, previous = [], input_dim
        for width in hidden_sizes:
            layers.extend([nn.Linear(previous, width), nn.Tanh()])
            previous = width
        layers.append(nn.Linear(previous, 1))
        self.network = nn.Sequential(*layers)

    def forward(self, features):
        # Scalar per sample: [B], never [B,1].
        return self.network(features).squeeze(-1)


class DualVariable(nn.Module):
    def __init__(self, initial=1.0):
        super().__init__()
        self.value = nn.Parameter(torch.tensor(float(initial)))


class LabelReplay:
    """Bounded, class-balanced CPU storage; its RNG is checkpointed with it."""
    def __init__(self, capacity=8192, seed=0):
        if capacity < 2:
            raise ValueError("Replay capacity must be at least two")
        self.safe = deque(maxlen=capacity // 2)
        self.unsafe = deque(maxlen=capacity - capacity // 2)
        self.rng = np.random.default_rng(seed)

    def add(self, features, labels):
        for sample, label in zip(features, labels, strict=True):
            if label == -1:
                self.safe.append(np.array(sample, np.float32, copy=True))
            elif label == 1:
                self.unsafe.append(np.array(sample, np.float32, copy=True))

    def sample(self, batch_size, device):
        result = []
        for pool in (self.safe, self.unsafe):
            if pool:
                indices = self.rng.integers(0, len(pool), size=max(1, batch_size // 2))
                result.append(torch.as_tensor(np.stack([pool[i] for i in indices]), device=device))
            else:
                result.append(None)
        return result


def separation_losses(barrier, safe, unsafe, epsilon):
    if epsilon <= 0:
        raise ValueError("A positive separation margin is required")
    zero = next(barrier.parameters()).sum() * 0
    feasible = torch.relu(epsilon + barrier(safe)).mean() if safe is not None else zero
    infeasible = torch.relu(epsilon - barrier(unsafe)).mean() if unsafe is not None else zero
    return feasible, infeasible


def horizon_pairs(states, next_states, episode_ends, horizons):
    """Pair each origin with actual successors, never across an episode reset.

    Input: [time, environment, feature]. The final physical transition is retained
    even when it ends an episode; only subsequent cross-reset pairs are excluded.
    """
    if states.shape != next_states.shape or episode_ends.shape != states.shape[:2]:
        raise ValueError("Inconsistent transition arrays")
    time, envs, dims = states.shape
    if horizons < 1:
        raise ValueError("horizons must be positive")
    successors = np.zeros((time, envs, horizons, dims), np.float32)
    valid = np.zeros((time, envs, horizons), bool)
    alive = np.ones((time, envs), bool)
    for n in range(1, horizons + 1):
        count = time - n + 1
        if count <= 0:
            break
        if n > 1:
            alive[:count] &= ~episode_ends[n - 2:time - 1]
        successors[:count, :, n - 1] = next_states[n - 1:]
        valid[:count, :, n - 1] = alive[:count]
    return states.reshape(-1, dims), successors.reshape(-1, horizons, dims), valid.reshape(-1, horizons)


def invariant_residuals(barrier, states, successors, valid, epsilon, contraction, discount):
    """One nonnegative residual per origin; normalize over available horizons.

    epsilon * sum_{k=0}^{n-1}(1-contraction)^k is the accumulated robust margin.
    This boundary normalization is explicit and differs from zero-padding all
    missing horizons in the 2023 reference implementation.
    """
    if not 0 < contraction < 1 or not 0 < discount <= 1:
        raise ValueError("Invalid barrier contraction/discount")
    batch, horizons, dims = successors.shape
    if states.shape != (batch, dims) or valid.shape != (batch, horizons):
        raise ValueError("Inconsistent horizon tensors")
    n = torch.arange(1, horizons + 1, device=states.device, dtype=states.dtype)
    factors = (1 - contraction) ** n
    margins = epsilon * (1 - factors) / contraction
    current = barrier(states)
    following = barrier(successors.reshape(-1, dims)).reshape(batch, horizons)
    violation = torch.relu(margins + following - current[:, None] * factors)
    weights = discount ** (n - 1) * valid.to(states.dtype)
    denominator = weights.sum(1)
    if torch.any(denominator <= 0):
        raise ValueError("Every origin must have an actual one-step successor")
    return (violation * weights).sum(1) / denominator
