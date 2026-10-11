"""Make a small portable source bundle, excluding experiments and nested Git repos."""
import argparse
from pathlib import Path
import tarfile

PROJECT = Path(__file__).resolve().parents[1]
INCLUDE = ["safe_hvac", "configs/safe_hvac", "tests/safe_hvac", "docs/safe_hvac",
           "scripts/bootstrap_references.py", "scripts/setup_safe_hvac_gpu.sh",
           "scripts/run_safe_hvac_pilot.sh", "scripts/package_safe_hvac.py",
           "references/sources.json", "patches/hvac_safe_rl_legacy.patch",
           "requirements-safe-hvac.txt", "pytest.ini", "README_SAFE_HVAC.md", ".gitignore"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="dist/safe-hvac-ready.tar.gz")
    args = parser.parse_args()
    output = Path(args.output).resolve()
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(output, "w:gz") as archive:
        for name in INCLUDE:
            path = PROJECT / name
            files = sorted(p for p in path.rglob("*") if p.is_file()) if path.is_dir() else [path]
            for item in files:
                if "__pycache__" in item.parts or item.suffix == ".pyc":
                    continue
                archive.add(item, arcname=str(Path("safe-hvac-ready") / item.relative_to(PROJECT)))
    print(f"Portable source bundle: {output} ({output.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
