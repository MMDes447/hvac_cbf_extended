"""Preserve actual physical successors alongside SB3's rollout data."""
from typing import NamedTuple
import numpy as np
from stable_baselines3.common.buffers import RolloutBuffer


class SafetySamples(NamedTuple):
    observations: object
    actions: object
    old_values: object
    old_log_prob: object
    advantages: object
    returns: object
    safety_residuals: object


class SafetyRolloutBuffer(RolloutBuffer):
    def __init__(self, *args, feature_dim, **kwargs):
        self.feature_dim = int(feature_dim)
        super().__init__(*args, **kwargs)

    def reset(self):
        super().reset()
        shape = (self.buffer_size, self.n_envs)
        self.physical_states = np.zeros((*shape, self.feature_dim), np.float32)
        self.physical_next_states = np.zeros_like(self.physical_states)
        self.labels = np.zeros(shape, np.int8)
        self.next_labels = np.zeros(shape, np.int8)
        self.episode_ends = np.zeros(shape, bool)
        self.safety_residuals = np.zeros(shape, np.float32)
        self.pending = None
        self.safety_flattened = False

    def set_transitions(self, infos):
        self.pending = [info["safe_hvac_transition"] for info in infos]

    def add(self, *args, **kwargs):
        if self.pending is None or len(self.pending) != self.n_envs:
            raise RuntimeError("Every collected action needs its actual physical successor")
        for env, item in enumerate(self.pending):
            for key, target in (("state", self.physical_states),
                                ("next_state", self.physical_next_states)):
                value = np.asarray(item[key], np.float32)
                if value.shape != (self.feature_dim,) or not np.isfinite(value).all():
                    raise ValueError("Invalid physical transition features")
                target[self.pos, env] = value
            self.labels[self.pos, env] = item["label"]
            self.next_labels[self.pos, env] = item["next_label"]
            self.episode_ends[self.pos, env] = item["episode_end"]
        self.pending = None
        super().add(*args, **kwargs)

    def get(self, batch_size=None):
        if not self.safety_flattened:
            self.safety_residuals = self.swap_and_flatten(self.safety_residuals).reshape(-1)
            self.safety_flattened = True
        yield from super().get(batch_size)

    def _get_samples(self, batch_inds, env=None):
        ordinary = super()._get_samples(batch_inds, env)
        return SafetySamples(*ordinary, self.to_torch(self.safety_residuals[batch_inds]))
