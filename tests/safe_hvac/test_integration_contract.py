"""Numerical fixtures only: no EnergyPlus, BOPTEST, or experiment training."""
from types import SimpleNamespace
import numpy as np
import pytest
import torch
import gymnasium as gym
from stable_baselines3.common.logger import configure
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from safe_hvac.algorithm import BarrierPPO
from safe_hvac.buffer import SafetyRolloutBuffer
from safe_hvac.checkpoints import save_bundle, preserve_rng
from safe_hvac.config import DEFAULTS, load_config
from safe_hvac.environment import SafetyRecordWrapper
from safe_hvac.legacy import PPOAgent1_epch
from safe_hvac.safety import SafetySpec
from safe_hvac.storage import RunStore, atomic_json


class ContractEnv(gym.Env):
    observation_variables = ["month", "day_of_month", "hour", "temp", "HVAC_electricity_demand_rate"]
    observation_space = gym.spaces.Box(-1e6, 1e6, (5,), dtype=np.float32)
    action_space = gym.spaces.Box(np.array([12, 23.25], np.float32), np.array([23.25, 30], np.float32))

    def __init__(self):
        self.model = SimpleNamespace(timestep_per_episode=4)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.step_count = 0
        return np.array([1, 1, 0, 22, 1000], np.float32), {"time_elapsed(hours)": .25}

    def step(self, action):
        self.last_action = action.copy()
        self.step_count += 1
        return (np.array([1, 1, self.step_count / 4, 20, 1000], np.float32), -1., False, False,
                {"time_elapsed(hours)": .25 * (self.step_count + 1)})


def test_wrapper_physical_records_terminal_pair_action_scaling():
    env = SafetyRecordWrapper(ContractEnv(), SafetySpec(["temp"]))
    obs, _ = env.reset()
    previous = env.current_features.copy()
    for i in range(3):
        obs, reward, terminated, truncated, info = env.step(np.zeros(2, np.float32))
        np.testing.assert_array_equal(info["safe_hvac_transition"]["state"], previous)
        previous = info["safe_hvac_transition"]["next_state"].copy()
        assert info["safe_hvac_record"]["energy_kwh"] == .25
        assert info["safe_hvac_record"]["degree_hours"] == .25
        assert truncated == (i == 2)
    np.testing.assert_allclose(env.unwrapped.last_action, [17.625, 26.625])
    assert info["safe_hvac_transition"]["episode_end"]
    assert info["safe_hvac_episode"]["energy_kwh"] == .75


def test_wrapper_rejects_stale_simulator_end():
    env = SafetyRecordWrapper(ContractEnv(), SafetySpec(["temp"]))
    env.reset()
    env.elapsed = .5
    with pytest.raises(RuntimeError, match="time advance"):
        env.step(np.zeros(2, np.float32))


def test_buffer_matching_after_sb3_environment_major_shuffle():
    buffer = SafetyRolloutBuffer(2, gym.spaces.Box(-1, 1, (1,), dtype=np.float32),
                                 gym.spaces.Discrete(2), device="cpu", n_envs=2, feature_dim=1)
    for time in range(2):
        infos = [{"safe_hvac_transition": dict(state=[10 * env + time], next_state=[10 * env + time + 1],
                                              label=-1, next_label=1, episode_end=time == 1)} for env in range(2)]
        buffer.set_transitions(infos)
        buffer.add(np.array([[time], [10 + time]], np.float32), np.zeros((2, 1)), np.zeros(2),
                   np.zeros(2), torch.zeros(2), torch.zeros(2))
    buffer.safety_residuals[:] = [[0, 10], [1, 11]]
    buffer.advantages[:] = 1
    buffer.returns[:] = 1
    for batch in buffer.get(2):
        torch.testing.assert_close(batch.observations[:, 0], batch.safety_residuals)
    np.testing.assert_array_equal(buffer.physical_next_states[:, :, 0], [[1, 11], [2, 12]])


def _fixture_model():
    vec = VecNormalize(DummyVecEnv([ContractEnv]), norm_reward=False)
    model = BarrierPPO("MlpPolicy", vec, n_steps=4, batch_size=2, n_epochs=1,
                       feature_dim=1, safety_config={**DEFAULTS["barrier"], "updates_per_rollout": 1,
                                                     "batch_size": 4, "horizons": 2, "max_dual": 1.001},
                       device="cpu", seed=0)
    model._logger = configure(format_strings=[])
    return model, vec


def _fill_update_fixture(model):
    """Inject a four-row dataset; this never calls collect_rollouts or learn."""
    buf = model.rollout_buffer
    buf.reset()
    with torch.no_grad():
        for i in range(4):
            obs = np.array([[1, 1, i, 22, 1000]], np.float32)
            actions, values, log_probs = model.policy(torch.as_tensor(obs))
            buf.set_transitions([{"safe_hvac_transition": dict(state=[float(i)], next_state=[i + .5],
                                                              label=-1 if i < 2 else 1,
                                                              next_label=-1 if i < 2 else 1,
                                                              episode_end=i in (1, 3))}])
            buf.add(obs, actions.numpy(), np.array([1.]), np.array([i in (0, 2)]), values, log_probs)
    buf.compute_returns_and_advantage(torch.tensor([0.]), np.array([True]))
    model.num_timesteps += 4


def test_optimizer_fixture_checkpoint_reload_preserves_all_state(tmp_path):
    model, vec = _fixture_model()
    _fill_update_fixture(model)
    reports = []
    model.reporter = lambda _, metrics: reports.append(metrics)
    model.train()  # Numerical optimizer regression, no environment interaction.
    assert reports[0]["train/policy_parameter_delta"] > 0
    assert reports[0]["train/actor_head_parameter_delta"] > 0
    assert reports[0]["safety/barrier_parameter_delta"] > 0
    model._dual_update(-100)
    assert 0 <= model.dual.value.item() <= 1.001
    save_bundle(model, vec, tmp_path / "bundle", {"fixture": True}, {})
    restored = BarrierPPO.load(tmp_path / "bundle/model.zip", env=vec, device="cpu")
    restored._logger = configure(format_strings=[])
    for attribute in ("policy", "barrier", "dual"):
        for a, b in zip(getattr(model, attribute).parameters(), getattr(restored, attribute).parameters(), strict=True):
            torch.testing.assert_close(a, b)
    for name in ("barrier_optimizer", "dual_optimizer"):
        assert len(getattr(restored, name).state) == len(getattr(model, name).state) > 0
    assert len(restored.policy.optimizer.state) == len(model.policy.optimizer.state) > 0
    assert len(restored.label_replay.safe) == len(model.label_replay.safe)
    assert restored.dual_optimizer.param_groups[0]["params"][0] is restored.dual.value
    _fill_update_fixture(restored)
    restored.train()
    assert restored.num_timesteps == 8
    vec.close()


def test_pretraining_fixture_does_not_update_policy_or_accumulate_rollouts():
    model, vec = _fixture_model()
    model.barrier_pretrain_steps = 4
    _fill_update_fixture(model)
    before = [p.detach().clone() for p in model.policy.parameters()]
    model.train()
    assert model.policy_updates == 0 and model.barrier_updates == 1
    for a, b in zip(model.policy.parameters(), before, strict=True):
        torch.testing.assert_close(a, b)
    _fill_update_fixture(model)
    model.train()
    assert model.policy_updates > 0 and model.barrier_updates == 2
    vec.close()


def test_legacy_keeps_terminal_pair_and_dual_optimizer_binding(tmp_path):
    agent = PPOAgent1_epch(2, 1, batch_size=2, n_epochs=1, barrier_updates=1,
                          checkpoint_dir=tmp_path, device="cpu", replay_capacity=4)
    for i in range(2):
        state = np.array([float(i)], np.float32)
        action, log_prob, value = agent.choose_action(state)
        agent.remember(state, action, log_prob, value, 1., i == 1, state + 1,
                       terminated=i == 1)
    assert len(agent.pairs) == 2 and agent.memory.records[-1]["next_state"].item() == 2
    agent.learn_barrier()
    agent.learn()
    assert len(agent.memory) == 0 and agent.global_step > 0
    agent.save_models()
    restored = PPOAgent1_epch(2, 1, checkpoint_dir=tmp_path, device="cpu", replay_capacity=4)
    restored.load_models()
    assert restored.nu_optimizer.param_groups[0]["params"][0] is restored.nu
    torch.testing.assert_close(restored.nu, agent.nu)


def test_config_rejects_silent_overshoot_and_zero_policy_steps(tmp_path):
    path = tmp_path / "bad.yaml"
    path.write_text("total_timesteps: 513\n")
    with pytest.raises(ValueError, match="multiple"):
        load_config(path)
    path.write_text("barrier_pretrain_steps: 32768\n")
    with pytest.raises(ValueError, match="leave policy"):
        load_config(path)
    path.write_text("ppo:\n  invalid_option: 1\n")
    with pytest.raises(ValueError, match="Unknown"):
        load_config(path)


def test_run_records_cannot_overwrite_and_checkpoints_are_outside_simulator(tmp_path):
    store = RunStore(tmp_path / "run")
    store.write("steps", {"actual_next_state": np.array([1, 2])})
    store.close()
    with pytest.raises(FileExistsError):
        RunStore(tmp_path / "run")


def test_sb3_collection_captures_terminal_successor_before_autoreset():
    class Callback(BaseCallback):
        def _on_step(self):
            return True
    spec = SafetySpec(["temp"])
    vec = VecNormalize(DummyVecEnv([lambda: SafetyRecordWrapper(ContractEnv(), spec)]), norm_reward=False)
    model = BarrierPPO("MlpPolicy", vec, n_steps=4, batch_size=2, n_epochs=1,
                       feature_dim=spec.feature_dim, safety_config=DEFAULTS["barrier"], device="cpu", seed=1)
    model._logger = configure(format_strings=[])
    callback = Callback()
    model._setup_learn(4, callback=callback)
    # Known timeout bootstrap; policy collection itself stays unchanged.
    model.policy.predict_values = lambda obs: torch.full((len(obs), 1), 2.)
    assert model.collect_rollouts(vec, callback, model.rollout_buffer, n_rollout_steps=4)
    buf = model.rollout_buffer
    np.testing.assert_array_equal(buf.episode_ends[:, 0], [False, False, True, False])
    assert buf.rewards[2, 0] == pytest.approx(-1 + .99 * 2)
    # Next state of the ending transition is the physical 20 C observation;
    # current state of the following episode comes from its 22 C reset.
    assert buf.physical_next_states[2, 0, 0] == pytest.approx((20 - 22.5) / 1.5)
    assert buf.physical_states[3, 0, 0] == pytest.approx((22 - 22.5) / 1.5)
    assert model.policy_updates == 0  # collection test performs no training
    vec.close()


def test_evaluation_rng_context_preserves_training_sequence():
    import random
    random.seed(7)
    np.random.seed(7)
    torch.manual_seed(7)
    expected = (random.random(), np.random.rand(), torch.rand(1))
    random.seed(7)
    np.random.seed(7)
    torch.manual_seed(7)
    with preserve_rng():
        random.seed(999)
        np.random.seed(999)
        torch.manual_seed(999)
        torch.rand(30)
    assert random.random() == expected[0]
    assert np.random.rand() == expected[1]
    torch.testing.assert_close(torch.rand(1), expected[2])


def test_saved_evaluation_freezes_norm_and_reports_all_seeds(tmp_path, monkeypatch):
    from copy import deepcopy
    from safe_hvac import evaluate as module
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    vec = VecNormalize(DummyVecEnv([lambda: SafetyRecordWrapper(ContractEnv(), SafetySpec(["temp"]))]), norm_reward=False)
    vec.reset()
    vec.step(np.zeros((1, 2), np.float32))
    vec.save(bundle / "vecnormalize.pkl")
    before_mean = vec.obs_rms.mean.copy()
    before_count = vec.obs_rms.count
    vec.close()
    loaded = []
    def make(config, seed, directory, name, normalization_path, training):
        other = VecNormalize.load(normalization_path, DummyVecEnv([lambda: SafetyRecordWrapper(ContractEnv(), SafetySpec(["temp"]))]))
        other.training, other.norm_reward = training, False
        loaded.append(other)
        return other
    class Model:
        @classmethod
        def load(cls, path, device):
            assert path == bundle / "model.zip"
            return cls()
        def predict(self, obs, deterministic):
            return np.zeros((1, 2), np.float32), None
    monkeypatch.setattr(module, "make_vec_env", make)
    monkeypatch.setattr(module, "model_class", lambda config: Model)
    config = deepcopy(DEFAULTS)
    config["evaluation_seeds"] = [101, 102]
    store = RunStore(tmp_path / "evaluation")
    results = module.evaluate_bundle(bundle, store, config, device="cpu")
    store.close()
    assert [r["seed"] for r in results] == [101, 102]
    assert all(r["normalization_frozen"] and r["steps"] == 3 for r in results)
    for other in loaded:
        np.testing.assert_array_equal(other.obs_rms.mean, before_mean)
        assert other.obs_rms.count == before_count
