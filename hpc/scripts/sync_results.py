#!/usr/bin/env python3
"""
Sync results from UA HPC to local machine.

- Checks for running jobs before syncing
- Downloads via rsync through filexfer
- Optionally deletes from HPC after transfer (default: yes)
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path


def load_config(path: str) -> dict:
    with open(path) as f:
        cfg = json.load(f)
    for key in ("netid", "group"):
        if key not in cfg or cfg[key].startswith("YOUR"):
            print(f"ERROR: Set '{key}' in {path}")
            sys.exit(1)
    return cfg


def check_jobs(netid: str) -> bool:
    """Check for running/pending jobs on HPC."""
    print("Checking for running jobs...")
    cmd = [
        "ssh", f"{netid}@hpc.arizona.edu",
        f"ssh shell.hpc.arizona.edu 'squeue -r -u {netid} 2>/dev/null || echo NO_SQUEUE'"
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        output = result.stdout.strip()
        if "NO_SQUEUE" in output or not output:
            print("  Could not check (VPN required)")
            return False
        lines = [l for l in output.split("\n") if l.strip()]
        if len(lines) <= 1:
            print("  No running jobs.")
            return False
        print(f"  {len(lines) - 1} job(s) found:")
        for line in lines:
            print(f"    {line}")
        return True
    except (subprocess.TimeoutExpired, Exception) as e:
        print(f"  Could not check: {e}")
        return False


def list_remote(netid: str, results_path: str) -> bool:
    """List result directories on HPC."""
    print(f"\nResults on cluster ({results_path}):")
    host = f"{netid}@filexfer.hpc.arizona.edu"
    cmd = ["ssh", host, f"ls -la {results_path}/ 2>/dev/null || echo EMPTY"]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        output = result.stdout.strip()
        if "EMPTY" in output or not output:
            print("  None found.")
            return False
        print(output)
        return True
    except Exception as e:
        print(f"  Error: {e}")
        return False


def sync(netid: str, results_path: str, local_dir: str,
         keep_remote: bool = False, dry_run: bool = False):
    """Download results via rsync."""
    host = f"{netid}@filexfer.hpc.arizona.edu"
    remote = f"{host}:{results_path}/"
    Path(local_dir).mkdir(parents=True, exist_ok=True)

    cmd = ["rsync", "-ravz", "--progress", "--partial"]
    if not keep_remote:
        cmd.append("--remove-source-files")
    if dry_run:
        cmd.append("--dry-run")
    cmd.extend([remote, f"{local_dir}/"])

    cleanup = " (deleting from HPC after)" if not keep_remote and not dry_run else ""
    print(f"\nSyncing{cleanup}...")
    print(f"  {remote} → {local_dir}/")

    result = subprocess.run(cmd, timeout=7200)
    if result.returncode == 0:
        print("\nSync complete.")
        if not keep_remote and not dry_run:
            subprocess.run(
                ["ssh", host, f"find {results_path} -type d -empty -delete 2>/dev/null"],
                capture_output=True, timeout=30,
            )
    else:
        print(f"\nFailed (code {result.returncode})", file=sys.stderr)
        sys.exit(result.returncode)


def main():
    parser = argparse.ArgumentParser(description="Sync results from UA HPC")
    parser.add_argument("--config", default="hpc/config/myconfig.json")
    parser.add_argument("--local-dir", default="results")
    parser.add_argument("--list-only", action="store_true")
    parser.add_argument("--keep-remote", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config)
    netid = cfg["netid"]
    results_path = cfg.get("results_path", f"/groups/{cfg['group']}/results")

    has_jobs = check_jobs(netid)
    has_results = list_remote(netid, results_path)

    if args.list_only:
        return
    if not has_results:
        print("\nNothing to sync.")
        return
    if has_jobs:
        print("\nWARNING: Jobs still running. Results may be incomplete.")
        resp = input("Continue? [y/N] ")
        if resp.lower() != "y":
            return

    sync(netid, results_path, args.local_dir, args.keep_remote, args.dry_run)


if __name__ == "__main__":
    main()
