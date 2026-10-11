"""Small, independently testable numerical objectives."""

from __future__ import annotations

import numpy as np
import torch


def compute_gae(rewards, values, next_values, terminated, episode_ends,
                gamma=0.99, gae_lambda=0.95):
    """GAE with separate bootstrap and episode-boundary masks.

    Arrays are chronological [time] or [time, environment]. A truncation
    bootstraps from its FINAL observation, but never from the reset observation.
    """
    arrays = [np.asarray(x) for x in
              (rewards, values, next_values, terminated, episode_ends)]
    if any(x.shape != arrays[0].shape for x in arrays):
        raise ValueError("All GAE arrays must have exactly the same shape")
    if arrays[0].ndim not in (1, 2):
        raise ValueError("GAE expects [time] or [time, environment]")
    r, v, nv = (x.astype(np.float64) for x in arrays[:3])
    terminal, ends = (x.astype(bool) for x in arrays[3:])
    if np.any(terminal & ~ends):
        raise ValueError("A terminated transition must also end its episode")
    if not all(np.isfinite(x).all() for x in (r, v, nv)):
        raise ValueError("Nonfinite rollout data")
    advantage = np.zeros_like(r)
    following = np.zeros(r.shape[1:] or (), dtype=np.float64)
    for t in range(len(r) - 1, -1, -1):
        delta = r[t] + gamma * (~terminal[t]) * nv[t] - v[t]
        following = delta + gamma * gae_lambda * (~ends[t]) * following
        advantage[t] = following
    return advantage.astype(np.float32), (advantage + v).astype(np.float32)


def paired_safety_loss(ratio: torch.Tensor, residual: torch.Tensor):
    """Require one frozen safety assessment for each recorded action."""
    if ratio.ndim != 1 or residual.ndim != 1 or ratio.shape != residual.shape:
        raise ValueError(f"Safety vectors must match: {ratio.shape}, {residual.shape}")
    if not torch.isfinite(ratio).all() or not torch.isfinite(residual).all():
        raise FloatingPointError("Nonfinite safety surrogate")
    return (ratio * residual.detach()).mean()
