"""Shared, corrected BOPTEST entry points (optional legacy dependency required)."""
import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
import random
import numpy as np
import torch

from .legacy import PPOAgent1_epch
from .storage import RunStore, atomic_json


def train(factory, variant="standard", n_episodes=None, random_episodes=True,
          episode_length=48, step_period=1800, barrier_only_episodes=None,
          seed=123456, output=None, url="https://api.boptest.net"):
    n_episodes = n_episodes if n_episodes is not None else (40 if variant == "pretrained" else 20)
    pretrain = barrier_only_episodes if barrier_only_episodes is not None else (20 if variant == "pretrained" else 0)
    if not 0 <= pretrain < n_episodes:
        raise ValueError("Barrier-only episodes must leave at least one policy-training episode")
    if episode_length <= 0 or step_period <= 0:
        raise ValueError("Episode duration (hours) and step period (seconds) must be positive")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    output = output or f"results/boptest/{variant}-{datetime.now(timezone.utc):%Y%m%dT%H%M%S}"
    store = RunStore(output)
    env = None
    try:
        atomic_json(store.root / "config.json", dict(variant=variant, episodes=n_episodes,
                    barrier_only_episodes=pretrain, random_episodes=random_episodes,
                    episode_hours=episode_length, step_period_seconds=step_period, seed=seed, url=url))
        env = factory(url, "bestest_hydronic_heat_pump", RANDOM_EPISODES=random_episodes,
                      LEN=episode_length, step_period=step_period)
        agent = PPOAgent1_epch(env.action_space.n, input_dims=env.observation_space.shape,
                              batch_size=6, n_epochs=10, occupancy_aware=variant == "occupancy",
                              checkpoint_dir=store.root / "final")
        interactions = 0
        for episode in range(n_episodes):
            observation, _ = env.reset(seed=seed + episode)
            done, score = False, 0.
            while not done:
                action, log_prob, value = agent.choose_action(observation)
                following, reward, terminated, truncated, info = env.step(action)
                done = bool(terminated or truncated)
                agent.remember(observation, action, log_prob, value, reward, done, following,
                               terminated=terminated, truncated=truncated)
                interactions += 1
                score += float(reward)
                store.write("steps", dict(timestep=interactions, episode=episode + 1, state=observation,
                                          next_state=following, action=action, reward=reward,
                                          terminated=terminated, truncated=truncated,
                                          unsafe_after_action=agent.is_unsafe(following)))
                if len(agent.memory) >= 48 or done:
                    barrier_loss = agent.learn_barrier()
                    if episode < pretrain:
                        agent.memory.clear_memory()  # No off-policy pretraining backlog.
                        metrics = None
                    else:
                        metrics = agent.learn()
                    store.write("updates", dict(timestep=interactions, barrier_loss=barrier_loss,
                                                policy=metrics, dual=float(agent.nu.detach())))
                observation = following
            store.write("episodes", dict(episode=episode + 1, score=score, interactions=interactions))
            print(f"Episode {episode + 1}: score={score:.3f}, interactions={interactions}")
        # Final model is selected a priori, including partial last-rollout updates.
        agent.save_models()
        atomic_json(store.root / "status.json", dict(status="complete", interactions=interactions,
                                                    policy_optimizer_steps=agent.global_step,
                                                    barrier_optimizer_steps=agent.barrier_steps))
        return store.root
    finally:
        if env is not None:
            env.close()
        store.close()


def evaluate(factory, variant, bundle, output, n_eval_episodes=3, seed=6767,
             url="https://api.boptest.net"):
    settings = json.loads((Path(bundle).parent / "config.json").read_text())
    if settings["variant"] != variant:
        raise ValueError("Evaluation variant differs from the training configuration")
    store = RunStore(output)
    env = None
    try:
        env = factory(url, "bestest_hydronic_heat_pump", RANDOM_EPISODES=True,
                      LEN=settings["episode_hours"], step_period=settings["step_period_seconds"])
        agent = PPOAgent1_epch(env.action_space.n, env.observation_space.shape,
                              occupancy_aware=variant == "occupancy", checkpoint_dir=bundle)
        agent.load_models()
        results = []
        for episode in range(n_eval_episodes):
            state, _ = env.reset(seed=seed + episode)
            done, score, violations, steps = False, 0., 0, 0
            while not done:
                action, _, _ = agent.choose_action(state, deterministic=True)
                following, reward, terminated, truncated, info = env.step(action)
                done = bool(terminated or truncated)
                violations += int(agent.is_unsafe(following))
                score += float(reward)
                steps += 1
                store.write("steps", dict(episode=episode + 1, state=state, next_state=following,
                                          reward=reward, action=action, terminated=terminated, truncated=truncated))
                state = following
            results.append(dict(episode=episode + 1, seed=seed + episode, score=score,
                                violation_rate=violations / steps, steps=steps))
        atomic_json(store.root / "evaluation.json", results)
        return results
    finally:
        if env is not None:
            env.close()
        store.close()


def main(factory, variant, evaluation=False):
    parser = argparse.ArgumentParser(description="Corrected legacy BOPTEST agent; use safe_hvac.train for Sinergym")
    parser.add_argument("--output", default=None)
    parser.add_argument("--seed", type=int, default=6767 if evaluation else 123456)
    parser.add_argument("--url", default="https://api.boptest.net")
    if evaluation:
        parser.add_argument("--bundle", required=True, help="Directory containing corrected agent.pt")
        parser.add_argument("--episodes", type=int, default=3)
    else:
        parser.add_argument("--episodes", type=int, default=40 if variant == "pretrained" else 20)
        parser.add_argument("--random", action=argparse.BooleanOptionalAction, default=True)
        parser.add_argument("--length", type=float, default=48, help="Episode duration in HOURS")
        parser.add_argument("--step_period", type=int, default=1800)
        parser.add_argument("--barrier_only", type=int, default=20 if variant == "pretrained" else 0)
    args = parser.parse_args()
    if evaluation:
        output = args.output or f"results/boptest/eval-{variant}-{datetime.now(timezone.utc):%Y%m%dT%H%M%S}"
        evaluate(factory, variant, args.bundle, output, args.episodes, args.seed, args.url)
    else:
        train(factory, variant, args.episodes, args.random, args.length, args.step_period,
              args.barrier_only, args.seed, args.output, args.url)
