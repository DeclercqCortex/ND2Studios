#!/usr/bin/env python3
"""Upload project code to UA HPC cluster via rsync."""

import argparse
import json
import subprocess
import sys
from pathlib import Path

EXCLUDE = [
    "__pycache__", ".git", "*.pyc", "results/", "slurm_logs/",
    "*.tar.gz", ".DS_Store", "venv/", ".venv/", "env/",
]


def load_config(path: str) -> dict:
    with open(path) as f:
        cfg = json.load(f)
    for key in ("netid", "group", "repo_path"):
        if key not in cfg or cfg[key].startswith("YOUR"):
            print(f"ERROR: Set '{key}' in {path}")
            sys.exit(1)
    return cfg


def main():
    parser = argparse.ArgumentParser(description="Upload project to UA HPC")
    parser.add_argument("--config", default="hpc/config/myconfig.json")
    parser.add_argument("--local-dir", default=".")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config)
    host = f"{cfg['netid']}@filexfer.hpc.arizona.edu"
    remote = f"{host}:{cfg['repo_path']}/"

    cmd = ["rsync", "-ravz", "--progress"]
    if args.dry_run:
        cmd.append("--dry-run")
    for exc in EXCLUDE:
        cmd.extend(["--exclude", exc])
    cmd.extend([f"{args.local_dir}/", remote])

    print(f"Syncing to {remote}")
    result = subprocess.run(cmd)
    if result.returncode == 0:
        print("\nSync complete.")
    else:
        print(f"\nFailed (code {result.returncode})", file=sys.stderr)
        sys.exit(result.returncode)


if __name__ == "__main__":
    main()
