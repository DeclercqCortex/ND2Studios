#!/usr/bin/env python3
"""
Auto-Version Push Tool
======================

Automates the VSClaude version control workflow:
1. Detect current version from branch name
2. Snapshot pip dependencies → requirements.txt
3. Stage all changes and commit
4. Push current branch to origin (this IS the release)
5. Create next version branch
6. Push new branch to origin
7. Switch to new branch — ready for next feature

Usage:
    python3 scripts/version_push.py
    python3 scripts/version_push.py -m "Add contact network model"
    python3 scripts/version_push.py --major        # 2.9 → 3.0
    python3 scripts/version_push.py --no-freeze    # skip pip freeze
    python3 scripts/version_push.py --dry-run      # show what would happen
"""

import argparse
import re
import subprocess
import sys
from pathlib import Path


def run(cmd: str, check: bool = True, capture: bool = True, dry_run: bool = False) -> str:
    """Run a shell command and return stdout."""
    if dry_run:
        print(f"  [DRY RUN] {cmd}")
        return ""
    result = subprocess.run(
        cmd, shell=True, capture_output=capture, text=True
    )
    if check and result.returncode != 0:
        stderr = result.stderr.strip() if result.stderr else ""
        print(f"ERROR: Command failed: {cmd}")
        if stderr:
            print(f"  {stderr}")
        sys.exit(1)
    return result.stdout.strip() if result.stdout else ""


def get_current_version() -> tuple[str, int, int]:
    """Get current version from branch name. Returns (branch_name, major, minor)."""
    branch = run("git rev-parse --abbrev-ref HEAD")
    match = re.match(r"Version-(\d+)\.(\d+)", branch)
    if not match:
        print(f"ERROR: Current branch '{branch}' doesn't match 'Version-X.Y' pattern.")
        print("Switch to a version branch first: git checkout Version-X.Y")
        sys.exit(1)
    return branch, int(match.group(1)), int(match.group(2))


def next_version(major: int, minor: int, is_major_bump: bool) -> tuple[int, int]:
    """Calculate next version number."""
    if is_major_bump:
        return major + 1, 0
    return major, minor + 1


def pip_freeze(dry_run: bool = False):
    """Snapshot current pip dependencies to requirements.txt."""
    print("Freezing pip dependencies → requirements.txt")
    if dry_run:
        print("  [DRY RUN] pip freeze > requirements.txt")
        return
    try:
        deps = run("pip freeze", check=False)
        if deps:
            Path("requirements.txt").write_text(deps + "\n")
            print(f"  Captured {len(deps.splitlines())} packages")
        else:
            print("  No pip packages found (not in a venv?)")
    except Exception as e:
        print(f"  WARNING: pip freeze failed ({e}), skipping")


def check_uncommitted() -> bool:
    """Check if there are uncommitted changes."""
    status = run("git status --porcelain")
    return bool(status)


def check_remote_exists() -> bool:
    """Check if 'origin' remote is configured."""
    remotes = run("git remote", check=False)
    return "origin" in remotes.split()


def update_claude_md_version(major: int, minor: int, dry_run: bool = False):
    """Update the version string in CLAUDE.md if present."""
    claude_md = Path("CLAUDE.md")
    if not claude_md.exists():
        return
    content = claude_md.read_text()
    # Match pattern like "Current version: V2.3" or "**Current version: V2.3**"
    new_content = re.sub(
        r"(\*?\*?Current version:\s*V)\d+\.\d+(\*?\*?)",
        rf"\g<1>{major}.{minor}\2",
        content,
    )
    if new_content != content and not dry_run:
        claude_md.write_text(new_content)
        print(f"  Updated CLAUDE.md version → V{major}.{minor}")


def main():
    parser = argparse.ArgumentParser(
        description="Auto-version push: commit, push, create next branch"
    )
    parser.add_argument(
        "-m", "--message",
        type=str,
        default=None,
        help="Commit message (prompted if not provided)",
    )
    parser.add_argument(
        "--major",
        action="store_true",
        help="Major version bump (X.Y → (X+1).0 instead of X.(Y+1))",
    )
    parser.add_argument(
        "--no-freeze",
        action="store_true",
        help="Skip pip freeze step",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would happen without doing it",
    )
    args = parser.parse_args()
    dry = args.dry_run

    if dry:
        print("=== DRY RUN MODE ===\n")

    # 1. Get current version
    branch, major, minor = get_current_version()
    next_maj, next_min = next_version(major, minor, args.major)
    next_branch = f"Version-{next_maj}.{next_min}"

    print(f"Current: {branch} (V{major}.{minor})")
    print(f"Next:    {next_branch} (V{next_maj}.{next_min})")
    print()

    # 2. Check for changes
    if not check_uncommitted():
        print("No uncommitted changes. Nothing to push.")
        print("Make some changes first, then run this again.")
        sys.exit(0)

    # 3. Pip freeze
    if not args.no_freeze:
        pip_freeze(dry)
    print()

    # 4. Get commit message
    message = args.message
    if not message and not dry:
        print(f"Enter commit message for V{major}.{minor}:")
        message = input("> ").strip()
        if not message:
            print("ERROR: Commit message required.")
            sys.exit(1)
    elif not message:
        message = "[dry run commit message]"

    full_message = f"V{major}.{minor}: {message}"
    print(f"\nCommit message: {full_message}")

    # 5. Stage and commit
    print("\n--- Staging and committing ---")
    run("git add -A", dry_run=dry)
    # Use a temp file for the commit message to avoid shell escaping issues
    if not dry:
        msg_file = Path(".git/COMMIT_MSG_TMP")
        msg_file.write_text(full_message)
        run(f'git commit -F .git/COMMIT_MSG_TMP', dry_run=dry)
        msg_file.unlink(missing_ok=True)
    else:
        run(f'git commit -m "{full_message}"', dry_run=dry)

    # 6. Push current branch
    print(f"\n--- Pushing {branch} ---")
    if check_remote_exists():
        run(f"git push -u origin {branch}", dry_run=dry)
    else:
        print("  WARNING: No 'origin' remote. Skipping push.")
        print("  Add a remote: git remote add origin git@github.com:ORG/REPO.git")

    # 7. Create next branch
    print(f"\n--- Creating {next_branch} ---")
    run(f"git checkout -b {next_branch}", dry_run=dry)

    # 8. Update CLAUDE.md version in new branch
    update_claude_md_version(next_maj, next_min, dry)

    # 9. Push new branch
    if check_remote_exists():
        print(f"\n--- Pushing {next_branch} ---")
        if not dry:
            # Commit the version bump if CLAUDE.md was updated
            status = run("git status --porcelain")
            if status:
                run("git add -A")
                msg_file = Path(".git/COMMIT_MSG_TMP")
                msg_file.write_text(f"Start V{next_maj}.{next_min} development")
                run('git commit -F .git/COMMIT_MSG_TMP')
                msg_file.unlink(missing_ok=True)
        run(f"git push -u origin {next_branch}", dry_run=dry)

    print(f"\n{'=' * 50}")
    print(f"Released V{major}.{minor} on branch {branch}")
    print(f"Now on {next_branch} — ready for next feature")
    print(f"{'=' * 50}")


if __name__ == "__main__":
    main()
