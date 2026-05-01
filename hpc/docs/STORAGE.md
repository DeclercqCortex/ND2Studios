# Storage Strategy for UA HPC

## The Problem

HPC storage is limited, shared, and **not backed up**. The most common failure: filling `/home` (50 GB) breaks your login.

## Storage Tiers

| Location | Quota | Lifetime | Use for |
|----------|-------|----------|---------|
| `/home/uXX/netid` | **50 GB** | Account life | Code, configs |
| `/groups/pi_group` | **500 GB** | PI account life | **Results, environments** |
| `/xdisk/pi_group` | Up to 20 TB | **300 days** | Temporary large data |
| `/tmp` (compute node) | ~1 TB | **Job only** | Scratch |

## The Golden Rule

```
Write results to /groups → Sync to your laptop → Delete from /groups
```

## Avoiding /home Overflow

### Redirect caches

Add to `~/.bashrc` on HPC:
```bash
export PIP_CACHE_DIR=/groups/YOUR_GROUP/.pip_cache
export PYTHONUSERBASE=/groups/YOUR_GROUP/.local
```

### Put venvs in /groups (if /home is tight)
```bash
python3 -m venv --system-site-packages /groups/YOUR_GROUP/envs/my-env
```

### Check usage
```bash
uquota                    # overall quotas
du -hs $(ls -A ~)         # find space hogs
```

### Clean up
```bash
pip cache purge
rm -rf ~/.cache/pip
find ~ -name "__pycache__" -exec rm -rf {} +
```

## File Count Limits

The parallel filesystem is slow with many small files. Keep <100,000 files per directory.

For many small output files:
1. Tar on HPC: `tar czf results.tar.gz results/`
2. Transfer the archive
3. Extract locally

## Incremental Sync

For long sweeps, sync as tasks complete:
```bash
rsync -ravz --remove-source-files \
    uahpc-xfer:/groups/GROUP/results/ ./results/
ssh uahpc-xfer "find /groups/GROUP/results -type d -empty -delete"
```

## Transfer Methods

| Size | Method |
|------|--------|
| < 100 GB | rsync via `filexfer.hpc.arizona.edu` |
| > 100 GB | Globus (endpoint: "UA HPC Filesystems") |
| < 64 MB | Open OnDemand web upload |

**Always use `filexfer`** — the bastion has a 10 MB limit.
