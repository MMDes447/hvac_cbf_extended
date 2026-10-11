import numpy as np
import pytest
import torch

from safe_hvac.math import compute_gae, paired_safety_loss
from safe_hvac.safety import (BarrierNet, LabelReplay, SafetySpec, separation_losses,
                              horizon_pairs, invariant_residuals)
from safe_hvac.reward import PhysicalComfortReward


def test_pairing_loss_and_gradient_reject_broadcast():
    ratio = torch.tensor([.5, 1.5], requires_grad=True)
    violation = torch.tensor([0., 2.], requires_grad=True)
    loss = paired_safety_loss(ratio, violation)
    assert loss.item() == 1.5
    loss.backward()
    torch.testing.assert_close(ratio.grad, torch.tensor([0., 1.]))
    assert violation.grad is None
    with pytest.raises(ValueError):
        paired_safety_loss(ratio, violation[:, None])


def test_gae_last_reward_episode_boundary_and_timeout():
    args = ([1, 2, 3, 100], [4, 3, 2, 50], [3, 2, 0, 0], [False, False, True, True], [False, False, True, True])
    advantage, target = compute_gae(*args, gamma=.9, gae_lambda=.8)
    np.testing.assert_allclose(advantage, [.7944, 1.52, 1., 50.], rtol=1e-5)
    np.testing.assert_allclose(target, [4.7944, 4.52, 3., 100.], rtol=1e-5)
    advantage, target = compute_gae([3], [2], [6], [False], [True], gamma=.9)
    np.testing.assert_allclose(advantage, [6.4])
    np.testing.assert_allclose(target, [8.4])


def test_zero_barrier_does_not_satisfy_positive_margin():
    barrier = BarrierNet(1)
    for parameter in barrier.parameters():
        torch.nn.init.zeros_(parameter)
    x = torch.zeros((2, 1))
    safe, unsafe = separation_losses(barrier, x, x, .01)
    assert safe.item() == pytest.approx(.01)
    assert unsafe.item() == pytest.approx(.01)
    residual = invariant_residuals(barrier, x, x[:, None], torch.ones((2, 1), dtype=torch.bool), .01, .1, .99)
    torch.testing.assert_close(residual, torch.full((2,), .01))


def test_actual_horizon_successors_do_not_cross_resets():
    states = np.array([0, 1, 100, 101], np.float32).reshape(4, 1, 1)
    following = np.array([1, 2, 101, 102], np.float32).reshape(4, 1, 1)
    ends = np.array([False, True, False, True]).reshape(4, 1)
    origins, future, valid = horizon_pairs(states, following, ends, 3)
    np.testing.assert_array_equal(future[:, 0, 0], [1, 2, 101, 102])
    np.testing.assert_array_equal(valid, [[True, True, False], [True, False, False],
                                         [True, True, False], [True, False, False]])
    assert future[0, 1, 0] == 2
    assert future[2, 1, 0] == 102


def test_label_replay_is_bounded_and_handles_missing_class():
    replay = LabelReplay(8)
    replay.add(np.arange(100, dtype=np.float32)[:, None], np.full(100, -1))
    assert len(replay.safe) == 4 and len(replay.unsafe) == 0
    safe, unsafe = replay.sample(6, "cpu")
    assert safe.shape == (3, 1) and unsafe is None


def test_labels_use_physical_temperature_and_occupancy():
    spec = SafetySpec(["temp"], occupancy_variables=["people"])
    raw = dict(temp=18., people=0., month=1, day_of_month=1, hour=12,
               HVAC_electricity_demand_rate=1000.)
    result = spec.assess(raw)
    assert result["label"] == -1 and result["deviation_c"].item() == 0
    raw["people"] = 2
    result = spec.assess(raw)
    assert result["label"] == 1 and result["deviation_c"].item() == 3
    raw["temp"] = 21.1
    assert spec.assess(raw)["label"] == 0  # classification guard band, still report physical violations
    del raw["people"]
    with pytest.raises(ValueError):
        spec.assess(raw)


def test_reward_and_safety_use_the_same_occupancy_bounds():
    reward = PhysicalComfortReward(dict(temperature_variables=["temp"], occupancy_variables=["people"]))
    raw = dict(temp=18., people=0., month=1, day_of_month=1, hour=12,
               HVAC_electricity_demand_rate=1000.)
    value, terms = reward(raw)
    assert terms["comfort_term"] == 0 and value == pytest.approx(-.0025)
    raw["people"] = 2.
    value, terms = reward(raw)
    assert terms["comfort_term"] == pytest.approx(-2.25)
    assert value == pytest.approx(-2.2525)
