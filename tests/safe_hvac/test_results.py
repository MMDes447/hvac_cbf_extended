"""Validate comparison integrity using synthetic result files, not experiments."""
from copy import deepcopy
import json
import pytest
from safe_hvac.config import DEFAULTS
from safe_hvac.report import compare
from safe_hvac.storage import atomic_json


def _run(root, algorithm, seed, cost):
    root.mkdir()
    config = deepcopy(DEFAULTS)
    config.update(algorithm=algorithm, env_seed=seed)
    atomic_json(root / "config.json", config)
    atomic_json(root / "metadata.json", {"assets": {}, "software": {"packages": {"fixture": "1"}}})
    atomic_json(root / "status.json", {"status": "complete"})
    row = dict(seed=6767, score=-cost, steps=10, duration_hours=2.5,
               energy_kwh=1., mean_power_kw=.4, violation_rate=.1, degree_hours=.2,
               occupied_degree_hours=.2, max_deviation_c=.4, cumulative_cost=cost)
    atomic_json(root / "evaluation.json", {"per_seed": [dict(row, seed=seed) for seed in config["evaluation_seeds"]]})


def test_report_pairs_trained_models_and_excludes_duplicate_runs(tmp_path):
    _run(tmp_path / "ppo", "ppo", 101, 10)
    _run(tmp_path / "barrier", "barrier_ppo", 101, 8)
    summaries = compare([tmp_path / "ppo", tmp_path / "barrier"], tmp_path / "comparison")
    assert all(s["training_runs"] == 1 for s in summaries)
    paired = json.loads((tmp_path / "comparison/paired_differences.json").read_text())
    assert paired["pairs"][0]["cumulative_cost"] == -2
    with pytest.raises(ValueError, match="more than once"):
        compare([tmp_path / "ppo", tmp_path / "ppo"], tmp_path / "bad")


def test_report_rejects_different_task_or_unmatched_training_seed(tmp_path):
    _run(tmp_path / "ppo", "ppo", 101, 10)
    _run(tmp_path / "barrier", "barrier_ppo", 102, 8)
    with pytest.raises(ValueError, match="matched PPO"):
        compare([tmp_path / "ppo", tmp_path / "barrier"], tmp_path / "comparison")
    config = json.loads((tmp_path / "barrier/config.json").read_text())
    config["safety"]["winter_bounds"] = [18, 21]
    atomic_json(tmp_path / "barrier/config.json", config)
    with pytest.raises(ValueError, match="Runs differ"):
        compare([tmp_path / "ppo", tmp_path / "barrier"], tmp_path / "different-task")
