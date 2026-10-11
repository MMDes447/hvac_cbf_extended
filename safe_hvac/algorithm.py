"""PPO with a learned, detached transition-specific barrier penalty.

This is a policy-training constraint, not a runtime action shield or a formal
certificate. SB3 owns collection, timeout bootstrapping and GAE.
"""
from __future__ import annotations

import numpy as np
import torch
from torch.nn import functional as F
from stable_baselines3 import PPO
from stable_baselines3.common.utils import explained_variance
from gymnasium import spaces

from .buffer import SafetyRolloutBuffer
from .math import paired_safety_loss
from .safety import (BarrierNet, DualVariable, LabelReplay, separation_losses,
                     horizon_pairs, invariant_residuals)


def _snapshot(module):
    return [p.detach().clone() for p in module.parameters()]


def _delta(module, before):
    return float(torch.sqrt(sum((p.detach() - b).square().sum()
                               for p, b in zip(module.parameters(), before, strict=True))))


class ReportingPPO(PPO):
    """Unmodified SB3 PPO updates with an external, checkpointable run reporter."""
    reporter = None

    def _excluded_save_params(self):
        return super()._excluded_save_params() + ["reporter", "_collecting_safety"]

    def train(self):
        before = _snapshot(self.policy)
        actor_before = _snapshot(self.policy.action_net)
        critic_before = _snapshot(self.policy.value_net)
        super().train()
        metrics = {k: float(v) for k, v in self.logger.name_to_value.items()
                   if k.startswith("train/") and isinstance(v, (int, float, np.number))}
        for key, value in list(metrics.items()):
            if not np.isfinite(value):
                if key == "train/explained_variance":
                    metrics[key] = None  # Undefined for constant return targets.
                else:
                    raise FloatingPointError(f"Nonfinite PPO metric: {key}")
        metrics["train/policy_parameter_delta"] = _delta(self.policy, before)
        metrics["train/actor_head_parameter_delta"] = _delta(self.policy.action_net, actor_before)
        metrics["train/critic_head_parameter_delta"] = _delta(self.policy.value_net, critic_before)
        metrics["timesteps"] = self.num_timesteps
        if self.reporter:
            self.reporter(self, metrics)


class BarrierPPO(ReportingPPO):
    def __init__(self, *args, safety_config=None, feature_dim=None,
                 barrier_pretrain_steps=0, **kwargs):
        self.safety_config = dict(safety_config or {})
        self.feature_dim = feature_dim
        self.barrier_pretrain_steps = int(barrier_pretrain_steps)
        self._collecting_safety = False
        kwargs["rollout_buffer_class"] = SafetyRolloutBuffer
        kwargs["rollout_buffer_kwargs"] = {"feature_dim": feature_dim}
        super().__init__(*args, **kwargs)

    def _setup_model(self):
        super()._setup_model()
        if self.feature_dim is None:
            raise ValueError("Specify the physical barrier feature dimension")
        cfg = self.safety_config
        self.barrier = BarrierNet(self.feature_dim, cfg["hidden_sizes"]).to(self.device)
        self.barrier_optimizer = torch.optim.Adam(self.barrier.parameters(), lr=cfg["learning_rate"])
        self.dual = DualVariable(cfg["initial_dual"]).to(self.device)
        self.dual_optimizer = torch.optim.Adam(self.dual.parameters(), lr=cfg["dual_learning_rate"])
        if not hasattr(self, "label_replay"):
            self.label_replay = LabelReplay(cfg["replay_capacity"], self.seed or 0)
        self.barrier_updates = getattr(self, "barrier_updates", 0)
        self.policy_updates = getattr(self, "policy_updates", 0)
        self.dual_updates = getattr(self, "dual_updates", 0)

    def _get_torch_save_params(self):
        states, tensors = super()._get_torch_save_params()
        return states + ["barrier", "barrier_optimizer", "dual", "dual_optimizer"], tensors

    def collect_rollouts(self, *args, **kwargs):
        self._collecting_safety = True
        try:
            return super().collect_rollouts(*args, **kwargs)
        finally:
            self._collecting_safety = False

    def _update_info_buffer(self, infos, dones=None):
        if self._collecting_safety:
            self.rollout_buffer.set_transitions(infos)
        super()._update_info_buffer(infos, dones)

    def _barrier_update(self):
        cfg, buf = self.safety_config, self.rollout_buffer
        states, successors, valid = horizon_pairs(buf.physical_states, buf.physical_next_states,
                                                   buf.episode_ends, cfg["horizons"])
        self.label_replay.add(states, buf.labels.reshape(-1))
        self.label_replay.add(buf.physical_next_states.reshape(-1, self.feature_dim),
                              buf.next_labels.reshape(-1))
        states, successors, valid = [torch.as_tensor(x, device=self.device) for x in (states, successors, valid)]
        before = _snapshot(self.barrier)
        losses = []
        self.barrier.train()
        for _ in range(cfg["updates_per_rollout"]):
            indices = torch.randint(len(states), (min(cfg["batch_size"], len(states)),), device=self.device)
            safe, unsafe = self.label_replay.sample(cfg["batch_size"], self.device)
            feasible, infeasible = separation_losses(self.barrier, safe, unsafe, cfg["epsilon"])
            residual = invariant_residuals(self.barrier, states[indices], successors[indices], valid[indices],
                                           cfg["epsilon"], cfg["contraction"], cfg["discount"])
            # Anchor labelled certificate scores near ±anchor, keeping scale finite.
            zero = feasible * 0
            anchor = ((self.barrier(safe) + cfg["anchor"]).square().mean() if safe is not None else zero)
            anchor += ((self.barrier(unsafe) - cfg["anchor"]).square().mean() if unsafe is not None else zero)
            loss = feasible + infeasible + residual.mean() + cfg["anchor_weight"] * anchor
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite barrier loss")
            self.barrier_optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.barrier.parameters(), self.max_grad_norm)
            self.barrier_optimizer.step()
            self.barrier_updates += 1
            losses.append(float(loss.detach()))
        self.barrier.eval()
        with torch.no_grad():
            residual = invariant_residuals(self.barrier, states, successors, valid,
                                           cfg["epsilon"], cfg["contraction"], cfg["discount"])
            scores = self.barrier(states)
            labels = torch.as_tensor(buf.labels.reshape(-1), device=self.device)
            safe, unsafe = labels == -1, labels == 1
            metrics = dict(barrier_loss=np.mean(losses), barrier_parameter_delta=_delta(self.barrier, before),
                           residual_mean=float(residual.mean()), residual_max=float(residual.max()),
                           residual_positive_fraction=float((residual > 0).float().mean()),
                           safe_samples=int(safe.sum()), unsafe_samples=int(unsafe.sum()),
                           replay_safe=len(self.label_replay.safe), replay_unsafe=len(self.label_replay.unsafe))
            if safe.any():
                metrics["safe_margin_error_rate"] = float((scores[safe] > -cfg["epsilon"]).float().mean())
            if unsafe.any():
                metrics["unsafe_margin_error_rate"] = float((scores[unsafe] < cfg["epsilon"]).float().mean())
            policy_residual = residual
            if cfg["standardize_residual"]:
                policy_residual = (residual - residual.mean()) / (residual.std(unbiased=False) + 1e-8)
            # horizon_pairs is time-major. Buffer.get applies SB3's environment-major flattening.
            buf.safety_residuals[:] = policy_residual.cpu().numpy().reshape(buf.buffer_size, buf.n_envs)
        return metrics

    def _dual_update(self, mean_residual):
        loss = -self.dual.value * (float(mean_residual) - self.safety_config["violation_budget"])
        self.dual_optimizer.zero_grad()
        loss.backward()
        self.dual_optimizer.step()
        with torch.no_grad():
            self.dual.value.clamp_(0., self.safety_config["max_dual"])
        self.dual_updates += 1

    def train(self):
        cfg = self.safety_config
        barrier_metrics = self._barrier_update()
        before = _snapshot(self.policy)
        actor_before = _snapshot(self.policy.action_net)
        critic_before = _snapshot(self.policy.value_net)
        pretraining = self.num_timesteps <= self.barrier_pretrain_steps
        metrics = {"safety/" + k: v for k, v in barrier_metrics.items()}
        if not pretraining:
            self.policy.set_training_mode(True)
            self._update_learning_rate(self.policy.optimizer)
            clip_range = self.clip_range(self._current_progress_remaining)
            value_clip = (self.clip_range_vf(self._current_progress_remaining)
                          if self.clip_range_vf is not None else None)
            losses, kls, clip_fractions = [], [], []
            keep_training = True
            for _ in range(self.n_epochs):
                for batch in self.rollout_buffer.get(self.batch_size):
                    actions = batch.actions.long().flatten() if isinstance(self.action_space, spaces.Discrete) else batch.actions
                    if self.use_sde:
                        self.policy.reset_noise(self.batch_size)
                    values, log_prob, entropy = self.policy.evaluate_actions(batch.observations, actions)
                    values = values.flatten()
                    advantages = batch.advantages
                    if self.normalize_advantage and len(advantages) > 1:
                        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
                    ratio = torch.exp(log_prob - batch.old_log_prob)
                    actor_loss = -torch.min(advantages * ratio,
                                           advantages * torch.clamp(ratio, 1 - clip_range, 1 + clip_range)).mean()
                    safety_loss = paired_safety_loss(ratio, batch.safety_residuals)
                    combined = actor_loss + self.dual.value.detach() * safety_loss
                    if cfg["scale_actor_loss"]:
                        combined = combined / (1 + self.dual.value.detach())
                    predicted = (values if value_clip is None else batch.old_values +
                                 torch.clamp(values - batch.old_values, -value_clip, value_clip))
                    value_loss = F.mse_loss(batch.returns, predicted)
                    entropy_loss = log_prob.mean() if entropy is None else -entropy.mean()
                    loss = combined + self.ent_coef * entropy_loss + self.vf_coef * value_loss
                    if not torch.isfinite(loss):
                        raise FloatingPointError("Nonfinite PPO loss")
                    with torch.no_grad():
                        diff = log_prob - batch.old_log_prob
                        kl = float((torch.exp(diff) - 1 - diff).mean())
                        kls.append(kl)
                        clip_fractions.append(float((torch.abs(ratio - 1) > clip_range).float().mean()))
                    if self.target_kl is not None and kl > 1.5 * self.target_kl:
                        keep_training = False
                        break
                    self.policy.optimizer.zero_grad()
                    loss.backward()
                    norm = torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                    if not torch.isfinite(norm):
                        raise FloatingPointError("Nonfinite policy gradients")
                    self.policy.optimizer.step()
                    self.policy_updates += 1
                    losses.append([float(actor_loss.detach()), float(safety_loss.detach()),
                                   float(value_loss.detach()), float(entropy_loss.detach()), float(norm)])
                self._n_updates += 1
                if not keep_training:
                    break
            if not losses:
                raise RuntimeError("No policy optimizer update occurred; check target_kl")
            means = np.mean(losses, axis=0)
            metrics.update(dict(zip(["train/actor_loss", "train/safety_loss", "train/value_loss",
                                     "train/entropy_loss", "train/gradient_norm"], means, strict=True)))
            metrics["train/approx_kl"] = np.mean(kls)
            metrics["train/clip_fraction"] = np.mean(clip_fractions)
            metrics["train/explained_variance"] = explained_variance(self.rollout_buffer.values.flatten(),
                                                                     self.rollout_buffer.returns.flatten())
            self._dual_update(barrier_metrics["residual_mean"])
        metrics.update({"timesteps": self.num_timesteps, "train/pretraining": pretraining,
                        "train/policy_parameter_delta": _delta(self.policy, before),
                        "train/actor_head_parameter_delta": _delta(self.policy.action_net, actor_before),
                        "train/critic_head_parameter_delta": _delta(self.policy.value_net, critic_before),
                        "safety/dual": float(self.dual.value.detach()),
                        "safety/barrier_updates": self.barrier_updates,
                        "train/policy_optimizer_steps": self.policy_updates,
                        "safety/dual_updates": self.dual_updates})
        for key, value in metrics.items():
            if key != "train/explained_variance" and not np.isfinite(value):
                raise FloatingPointError(f"Nonfinite training metric: {key}")
            self.logger.record(key, value)
        if "train/explained_variance" in metrics and not np.isfinite(metrics["train/explained_variance"]):
            metrics["train/explained_variance"] = None
        if self.reporter:
            self.reporter(self, metrics)
