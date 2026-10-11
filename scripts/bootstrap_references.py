"""Clone pinned reference checkouts and apply the corrected legacy adapters.

No dependency installation or experiments. Existing work is never reset.
"""
import argparse
import json
from pathlib import Path
import subprocess

PROJECT = Path(__file__).resolve().parents[1]


def git(directory, *args, check=True):
    return subprocess.run(["git", "-C", str(directory), *args], check=check, capture_output=True, text=True)


def clone(name, source, target):
    if target.exists():
        result = git(target, "rev-parse", "HEAD")
        if result.stdout.strip() != source["commit"]:
            raise RuntimeError(f"Existing {target} has another commit; preserve it and choose an isolated checkout")
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(["git", "clone", "--no-checkout", source["url"], str(target)], check=True)
    git(target, "checkout", "--detach", source["commit"])
    print(f"Pinned {name}: {source['commit']}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--include-sinergym", action="store_true", help="Clone simulator source for inspection/installation reference")
    args = parser.parse_args()
    sources = json.loads((PROJECT / "references/sources.json").read_text())
    for name, target in (("srlnbc", PROJECT / "references/srlnbc"),
                         ("HVAC_SAFE_RL_", PROJECT / "HVAC_SAFE_RL_")):
        clone(name, sources[name], target)
    if args.include_sinergym:
        clone("sinergym", sources["sinergym"], PROJECT / "references/sinergym")
    target = PROJECT / "HVAC_SAFE_RL_"
    patch = PROJECT / "patches/hvac_safe_rl_legacy.patch"
    if git(target, "apply", "--reverse", "--check", str(patch), check=False).returncode == 0:
        print("Corrected legacy adapters already applied")
        return
    result = git(target, "apply", "--check", str(patch), check=False)
    if result.returncode:
        raise RuntimeError("Legacy checkout has conflicting changes; no files overwritten. " + result.stderr)
    git(target, "apply", str(patch))
    print("Applied corrected legacy adapters. New Sinergym package is independent of these clones.")


if __name__ == "__main__":
    main()
