"""Record actual software/assets and refuse the known incompatible simulator API."""
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path
import hashlib
import os
import platform
import subprocess
import sys
import torch


def software():
    versions = {}
    for name in ("sinergym", "stable-baselines3", "gymnasium", "numpy", "torch", "pandas", "PyYAML"):
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = None
    return dict(python=sys.version, platform=platform.platform(), packages=versions,
                cuda_available=torch.cuda.is_available(), torch_cuda=torch.version.cuda,
                gpu_names=[torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
                eplus_path=os.environ.get("EPLUS_PATH"))


def require_runtime(device="cuda"):
    versions = software()
    for name, expected in (("sinergym", "3.12.0"), ("stable-baselines3", "2.8.0")):
        if versions["packages"][name] != expected:
            raise RuntimeError(f"GPU runner requires {name}=={expected}; found {versions['packages'][name]}. "
                               "Use requirements-safe-hvac.txt in a separate environment.")
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but unavailable; install a compatible GPU PyTorch build")
    root = Path(os.environ.get("EPLUS_PATH", "/__missing_energyplus__"))
    executable = root / "energyplus"
    if not executable.is_file():
        raise RuntimeError("Set EPLUS_PATH to an EnergyPlus 25.1 installation")
    result = subprocess.run([str(executable), "--version"], text=True, capture_output=True, timeout=15)
    version = (result.stdout + result.stderr).strip()
    if result.returncode or "25.1" not in version:
        raise RuntimeError(f"Sinergym 3.12 assets require EnergyPlus 25.1; executable reported {version}")
    try:
        from pyenergyplus.api import EnergyPlusAPI
        EnergyPlusAPI()
    except Exception as error:
        raise RuntimeError("Cannot load EnergyPlus Python API; put EPLUS_PATH on PYTHONPATH") from error
    versions["energyplus"] = version
    return versions


def _git_state(directory):
    if not (Path(directory) / ".git").exists():
        return None
    def command(args):
        result = subprocess.run(["git", "-C", str(directory), *args], capture_output=True, text=True, timeout=10)
        return result.stdout.strip() if result.returncode == 0 else None
    return dict(commit=command(["rev-parse", "HEAD"]), status=command(["status", "--short"]))


def hash_file(path):
    path = Path(path)
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return dict(name=path.name, sha256=digest.hexdigest(), bytes=path.stat().st_size)


def manifest(config, vec_env=None, runtime=None):
    project = Path(__file__).resolve().parents[1]
    # A code digest is still useful when this workspace has no usable Git checkout.
    code = {str(p.relative_to(project)): hash_file(p)["sha256"]
            for p in sorted((project / "safe_hvac").glob("*.py"))}
    result = dict(created_utc=datetime.now(timezone.utc).isoformat(), software=runtime or software(),
                  situation_id=config["situation_id"], study_stage=config["study_stage"],
                  source_git=_git_state(project), code_sha256=code,
                  reference_lock=__import__("json").loads((project / "references/sources.json").read_text()),
                  seeds={key: config[key] for key in ("agent_seed", "env_seed", "evaluation_seeds")},
                  measured_temperature_variables=config["safety"]["temperature_variables"],
                  safety_scope="measured zones only; learned penalty, no runtime shield or formal certificate",
                  selection="final checkpoint; all evaluation seeds reported; no best-episode selection")
    if vec_env is not None:
        raw = vec_env.venv.envs[0].unwrapped
        result["assets"] = {key: hash_file(getattr(raw, key)) for key in ("building_path", "weather_path")}
        result["observation_variables"] = list(raw.observation_variables)
        result["native_action_low"] = raw.action_space.low.tolist()
        result["native_action_high"] = raw.action_space.high.tolist()
        result["physical_episode_steps"] = raw.model.timestep_per_episode - 1
        result["weather_variability"] = raw.default_options.get("weather_variability")
        result["physical_variable_sources"] = raw.variables
    return result
