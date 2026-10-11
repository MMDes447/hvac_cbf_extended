"""Read-only GPU host checks; does not reset an environment or train anything."""
import argparse
import json
from .config import load_config
from .provenance import require_runtime, software


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/safe_hvac/smoke_s01.yaml")
    args = parser.parse_args()
    config = load_config(args.config)
    print(json.dumps(software(), indent=2))
    require_runtime(config["device"])
    import gymnasium as gym
    import sinergym
    registration = gym.spec(config["environment"]["id"])
    observed = list(registration.kwargs.get("time_variables", [])) + list(registration.kwargs.get("variables", {})) + list(registration.kwargs.get("meters", {}))
    from .safety import SafetySpec
    missing = set(SafetySpec(**config["safety"]).required_variables()) - set(observed)
    if missing:
        raise ValueError(f"Environment registration lacks: {sorted(missing)}")
    print(f"Ready: {registration.id}; software, CUDA, EnergyPlus and registered variables checked.")


if __name__ == "__main__":
    main()
