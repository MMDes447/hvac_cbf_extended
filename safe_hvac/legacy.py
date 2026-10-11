"""Corrected categorical PPO API for the released BOPTEST scripts.

New Sinergym experiments use BarrierPPO. This adapter fixes the old entry points
and intentionally uses the same numerical helpers rather than three copies.
"""
from collections import deque
from pathlib import Path
import random
import numpy as np
import torch
from torch import nn

from .math import compute_gae, paired_safety_loss
from .safety import BarrierNet, DualVariable, LabelReplay, separation_losses


class _Network(nn.Module):
    def __init__(self, dims, outputs, learning_rate, device, categorical=False):
        super().__init__()
        self.network = nn.Sequential(nn.Linear(dims, 64), nn.Tanh(), nn.Linear(64, 64),
                                     nn.Tanh(), nn.Linear(64, outputs))
        self.device, self.categorical = torch.device(device), categorical
        self.to(self.device)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)

    def forward(self, states):
        output = self.network(states)
        return torch.distributions.Categorical(logits=output) if self.categorical else output.squeeze(-1)


class LegacyMemory:
    def __init__(self, capacity=4096):
        self.capacity = capacity
        self.records = []

    def __len__(self):
        return len(self.records)

    def clear_memory(self):
        self.records.clear()

    def add(self, record):
        if len(self.records) >= self.capacity:
            raise RuntimeError("PPO rollout exceeded its bound; flush learn() or clear pretraining memory")
        self.records.append(record)


class PPOAgent1_epch:
    def __init__(self, n_actions, input_dims, alpha=0.0003, batch_size=64,
                 n_epochs=10, gae_lambda=0.95, gamma=0.99, policy_clip=0.2,
                 occupancy_aware=False, device=None, checkpoint_dir="models",
                 replay_capacity=8192, rollout_capacity=4096,
                 barrier_updates=10, epsilon=0.01, lambda_param=0.1,
                 max_dual=10., violation_budget=0., label_guard_c=0.2):
        self.input_dims = int(np.prod(input_dims)) if isinstance(input_dims, (tuple, list)) else int(input_dims)
        if occupancy_aware and self.input_dims != 2:
            raise ValueError("Occupancy BOPTEST adapter requires temperature and occupancy observations")
        self.occupancy_aware = occupancy_aware
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.gamma, self.gae_lambda = gamma, gae_lambda
        self.batch_size, self.n_epochs, self.policy_clip = batch_size, n_epochs, policy_clip
        self.epsilon, self.lambda_param = epsilon, lambda_param
        if epsilon <= 0 or not 0 < lambda_param < 1:
            raise ValueError("Positive separation margin and valid contraction required")
        self.max_dual, self.violation_budget = max_dual, violation_budget
        self.label_guard_c = label_guard_c
        self.barrier_updates_per_flush = barrier_updates
        self.actor = _Network(self.input_dims, n_actions, alpha, self.device, categorical=True)
        self.critic = _Network(self.input_dims, 1, alpha, self.device)
        self.barrier = BarrierNet(self.input_dims).to(self.device)
        self.barrier.device = self.device
        self.barrier_optimizer = torch.optim.Adam(self.barrier.parameters(), lr=0.001)
        self.dual = DualVariable(1.).to(self.device)
        self.nu_optimizer = torch.optim.Adam(self.dual.parameters(), lr=0.001)
        self.memory = LegacyMemory(rollout_capacity)
        self.label_replay = LabelReplay(replay_capacity)
        self.pairs = deque(maxlen=replay_capacity)
        self.barrier_memory = {"initial_states": self.label_replay.safe, "unsafe_states": self.label_replay.unsafe,
                               "transitions": self.pairs}
        self.global_step = 0
        self.barrier_steps = 0
        self.checkpoint_dir = Path(checkpoint_dir)

    @property
    def nu(self):
        return self.dual.value

    def _state(self, observation):
        array = np.asarray(observation, np.float32).reshape(-1)
        if array.shape != (self.input_dims,) or not np.isfinite(array).all():
            raise ValueError(f"Expected {self.input_dims} finite observations")
        return array

    def _bounds(self, state):
        # BOPTEST temperature normalization is fixed: 18..21 C -> -1..1.
        # Occupancy normalization is fixed: 0..5 people -> -1..1.
        occupied = not self.occupancy_aware or (float(state[1]) + 1) * 2.5 > 0.5
        return (-1., 1.) if occupied else (-3., 7.)

    def _label(self, state):
        low, high = self._bounds(state)
        signed_c = max(low - state[0], state[0] - high) * 1.5
        return 1 if signed_c > self.label_guard_c else -1 if signed_c < -self.label_guard_c else 0

    def is_safe(self, state):
        low, high = self._bounds(state)
        return low <= state[0] <= high

    def is_unsafe(self, state):
        return not self.is_safe(state)

    def remember(self, state, action, probs, vals, reward, done, next_state,
                 *, terminated=None, truncated=False):
        state, next_state = self._state(state), self._state(next_state)
        if terminated is None:
            terminated = bool(done and not truncated)
        if bool(done) != bool(terminated or truncated):
            raise ValueError("done must equal terminated or truncated")
        with torch.no_grad():
            next_value = (0. if terminated else float(self.critic(torch.as_tensor(next_state[None], device=self.device))))
        record = dict(state=state.copy(), action=int(action), log_prob=float(probs), value=float(vals),
                      reward=float(reward), next_value=next_value, next_state=next_state.copy(),
                      terminated=bool(terminated), episode_end=bool(done))
        self.memory.add(record)
        self.pairs.append((state.copy(), next_state.copy()))
        self.label_replay.add([state, next_state], [self._label(state), self._label(next_state)])

    def barrier_feasible_loss(self, states):
        return torch.relu(self.epsilon + self.barrier(states)).mean()

    def barrier_infeasible_loss(self, states):
        return torch.relu(self.epsilon - self.barrier(states)).mean()

    def barrier_invariant_loss(self, states, next_states):
        if states.shape != next_states.shape:
            raise ValueError("Barrier states must be matched actual transitions")
        return torch.relu(self.epsilon + self.barrier(next_states) - (1 - self.lambda_param) * self.barrier(states))

    def learn_barrier(self):
        if not self.pairs:
            return None
        losses = []
        for _ in range(self.barrier_updates_per_flush):
            indices = np.random.randint(len(self.pairs), size=min(self.batch_size, len(self.pairs)))
            pairs = [self.pairs[i] for i in indices]
            states = torch.as_tensor(np.stack([p[0] for p in pairs]), device=self.device)
            following = torch.as_tensor(np.stack([p[1] for p in pairs]), device=self.device)
            safe, unsafe = self.label_replay.sample(self.batch_size, self.device)
            feasible, infeasible = separation_losses(self.barrier, safe, unsafe, self.epsilon)
            anchor = feasible * 0
            if safe is not None:
                anchor += (self.barrier(safe) + 0.1).square().mean()
            if unsafe is not None:
                anchor += (self.barrier(unsafe) - 0.1).square().mean()
            loss = feasible + infeasible + self.barrier_invariant_loss(states, following).mean() + 0.01 * anchor
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite legacy barrier loss")
            self.barrier_optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.barrier.parameters(), 0.5)
            self.barrier_optimizer.step()
            self.barrier_steps += 1
            losses.append(float(loss.detach()))
        return float(np.mean(losses))

    def choose_action(self, observation, deterministic=False):
        state = torch.as_tensor(self._state(observation)[None], device=self.device)
        with torch.no_grad():
            distribution = self.actor(state)
            action = distribution.probs.argmax(-1) if deterministic else distribution.sample()
            return int(action.item()), float(distribution.log_prob(action).item()), float(self.critic(state).item())

    def learn(self):
        records = self.memory.records
        if not records:
            return None
        advantage, targets = compute_gae(*[np.array([r[key] for r in records]) for key in
                                          ("reward", "value", "next_value", "terminated", "episode_end")],
                                         gamma=self.gamma, gae_lambda=self.gae_lambda)
        states = torch.as_tensor(np.stack([r["state"] for r in records]), device=self.device)
        following = torch.as_tensor(np.stack([r["next_state"] for r in records]), device=self.device)
        actions = torch.as_tensor([r["action"] for r in records], device=self.device, dtype=torch.long)
        old_probs = torch.as_tensor([r["log_prob"] for r in records], device=self.device)
        targets = torch.as_tensor(targets, device=self.device)
        advantages = torch.as_tensor(advantage, device=self.device)
        if len(records) > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std(unbiased=False) + 1e-8)
        with torch.no_grad():
            residual = self.barrier_invariant_loss(states, following)
        metrics = []
        for _ in range(self.n_epochs):
            indices = np.random.permutation(len(records))
            for start in range(0, len(indices), self.batch_size):
                batch = indices[start:start + self.batch_size]
                ratio = torch.exp(self.actor(states[batch]).log_prob(actions[batch]) - old_probs[batch])
                actor = -torch.min(ratio * advantages[batch], ratio.clamp(1 - self.policy_clip, 1 + self.policy_clip) * advantages[batch]).mean()
                safety = paired_safety_loss(ratio, residual[batch])
                combined = actor + self.nu.detach() * safety
                critic = (targets[batch] - self.critic(states[batch])).square().mean()
                if not torch.isfinite(combined + critic):
                    raise FloatingPointError("Nonfinite legacy PPO loss")
                self.actor.optimizer.zero_grad()
                self.critic.optimizer.zero_grad()
                combined.backward()
                critic.backward()
                torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)
                torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)
                self.actor.optimizer.step()
                self.critic.optimizer.step()
                self.global_step += 1
                metrics.append(dict(actor_loss=float(actor.detach()), critic_loss=float(critic.detach()),
                                    safety_loss=float(safety.detach())))
                if hasattr(self, "writer"):
                    for key, value in metrics[-1].items():
                        self.writer.add_scalar(f"Losses/{key}", value, self.global_step)
        self.nu_optimizer.zero_grad()
        (-self.nu * (residual.mean() - self.violation_budget)).backward()
        self.nu_optimizer.step()
        with torch.no_grad():
            self.nu.clamp_(0, self.max_dual)
        self.memory.clear_memory()
        return metrics[-1]

    def save_models(self):
        if len(self.memory):
            raise RuntimeError("Save only after flushing the current policy rollout")
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        payload = dict(actor=self.actor.state_dict(), critic=self.critic.state_dict(), barrier=self.barrier.state_dict(),
                       actor_optimizer=self.actor.optimizer.state_dict(), critic_optimizer=self.critic.optimizer.state_dict(),
                       barrier_optimizer=self.barrier_optimizer.state_dict(), dual=self.dual.state_dict(),
                       dual_optimizer=self.nu_optimizer.state_dict(), global_step=self.global_step,
                       barrier_steps=self.barrier_steps, replay=self.label_replay, pairs=self.pairs,
                       numpy_rng=np.random.get_state(), torch_rng=torch.get_rng_state(), python_rng=random.getstate(),
                       input_dims=self.input_dims, occupancy_aware=self.occupancy_aware,
                       hyperparameters={key: getattr(self, key) for key in ("gamma", "gae_lambda", "batch_size",
                           "n_epochs", "policy_clip", "epsilon", "lambda_param", "max_dual", "violation_budget",
                           "label_guard_c", "barrier_updates_per_flush")}, rollout_capacity=self.memory.capacity,
                       cuda_rng=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None)
        torch.save(payload, self.checkpoint_dir / "agent.pt.tmp")
        (self.checkpoint_dir / "agent.pt.tmp").replace(self.checkpoint_dir / "agent.pt")

    def load_models(self):
        state = torch.load(self.checkpoint_dir / "agent.pt", map_location=self.device, weights_only=False)
        if state["input_dims"] != self.input_dims or state["occupancy_aware"] != self.occupancy_aware:
            raise ValueError("Checkpoint observation/occupancy specification does not match")
        for key, target in (("actor", self.actor), ("critic", self.critic), ("barrier", self.barrier), ("dual", self.dual),
                            ("actor_optimizer", self.actor.optimizer), ("critic_optimizer", self.critic.optimizer),
                            ("barrier_optimizer", self.barrier_optimizer), ("dual_optimizer", self.nu_optimizer)):
            target.load_state_dict(state[key])
        self.global_step, self.barrier_steps = state["global_step"], state["barrier_steps"]
        for key, value in state["hyperparameters"].items():
            setattr(self, key, value)
        self.memory = LegacyMemory(state["rollout_capacity"])
        self.label_replay, self.pairs = state["replay"], state["pairs"]
        self.barrier_memory = {"initial_states": self.label_replay.safe, "unsafe_states": self.label_replay.unsafe,
                               "transitions": self.pairs}
        np.random.set_state(state["numpy_rng"])
        torch.set_rng_state(state["torch_rng"].cpu())
        random.setstate(state["python_rng"])
        if state["cuda_rng"] is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state_all([s.cpu() for s in state["cuda_rng"]])

    def evaluate_barrier(self, state):
        with torch.no_grad():
            return float(self.barrier(torch.as_tensor(self._state(state)[None], device=self.device)).item())
