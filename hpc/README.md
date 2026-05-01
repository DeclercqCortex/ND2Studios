# HPC Integration — University of Arizona (Puma)

Complete guide for running your computational project on the UA HPC cluster.

---

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [One-Time Local Setup](#one-time-local-setup)
3. [One-Time HPC Setup](#one-time-hpc-setup)
4. [Configuration](#configuration)
5. [Running Jobs](#running-jobs)
6. [Data Flow & Storage](#data-flow--storage)
7. [Syncing Results](#syncing-results)
8. [Parameter Sweeps](#parameter-sweeps)
9. [Monitoring](#monitoring)
10. [Quick Reference](#quick-reference)
11. [Common Gotchas](#common-gotchas)

---

## Prerequisites

- **UA NetID** with HPC access — request at [account.arizona.edu](https://account.arizona.edu)
- **PI/Group allocation** — your PI must have an active HPC allocation
- **Duo 2FA** enrolled
- **Cisco Secure Client VPN** — from [vpn.arizona.edu](https://vpn.arizona.edu) (required off-campus)

Test your access:
```bash
ssh YOUR_NETID@hpc.arizona.edu
# Type 'shell' → Puma login node
va        # check allocation hours
uquota    # check storage
```

---

## One-Time Local Setup

### SSH keys (passwordless transfers)

```bash
# Generate key if needed
ls ~/.ssh/id_ed25519 || ssh-keygen -t ed25519 -C "your_email@arizona.edu"

# Copy to both hosts (password + Duo once each)
ssh-copy-id YOUR_NETID@hpc.arizona.edu
ssh-copy-id YOUR_NETID@filexfer.hpc.arizona.edu
```

### SSH config shortcuts

Add to `~/.ssh/config`:
```
Host uahpc
    HostName hpc.arizona.edu
    User YOUR_NETID

Host uahpc-xfer
    HostName filexfer.hpc.arizona.edu
    User YOUR_NETID
```

---

## One-Time HPC Setup

### Python environment

**Must be done from an interactive compute session, not a login node.**

```bash
ssh uahpc
shell
interactive -a YOUR_GROUP -n 4 -t 1:00:00

# Create environment
module load python/3.11/3.11.4
python3 -m venv --system-site-packages ~/my-project-env
source ~/my-project-env/bin/activate
pip install --upgrade pip
pip install numpy scipy matplotlib
# Add your dependencies here

exit  # leave interactive session
```

**Tight on `/home` space?** Put the venv in `/groups`:
```bash
python3 -m venv --system-site-packages /groups/YOUR_GROUP/envs/my-project-env
```

---

## Configuration

Copy and edit the template:
```bash
cp hpc/config/template.json hpc/config/myconfig.json
```

```json
{
    "netid": "YOUR_NETID",
    "group": "YOUR_PI_GROUP",
    "cluster": "puma",
    "partition": "standard",
    "python_module": "python/3.11/3.11.4",
    "cpus": 4,
    "walltime": "04:00:00",
    "repo_path": "~/my-project",
    "venv_path": "~/my-project-env",
    "results_path": "/groups/YOUR_PI_GROUP/results",
    "max_concurrent": 10
}
```

| Field | Meaning | Notes |
|-------|---------|-------|
| `cpus` | CPUs per job | Each = 5 GB RAM on Puma. 4 CPUs = 20 GB |
| `partition` | `standard` (uses hours) or `windfall` (free, preemptible) | |
| `results_path` | **Must be `/groups`** not `/home` | Avoids 50 GB cap |
| `max_concurrent` | Max simultaneous array tasks | 10 is conservative |

---

## Running Jobs

### Upload code
```bash
python3 hpc/scripts/sync_to_hpc.py --config hpc/config/myconfig.json
```

### Submit a single job
```bash
# SSH to HPC
ssh uahpc && shell
cd ~/my-project
mkdir -p slurm_logs
sbatch hpc/scripts/run_single.slurm
```

### Check status
```bash
squeue --me
```

---

## Data Flow & Storage

**The golden rule:**
```
Code in /home  →  Results in /groups  →  Sync to local  →  Delete from HPC
```

| Location | Quota | Use for |
|----------|-------|---------|
| `/home` | **50 GB** | Code and configs only |
| `/groups` | **500 GB** | Results, environments, shared data |
| `/xdisk` | Up to 20 TB | Temporary large data (**300-day limit**) |

**Filling `/home` breaks your login.** Always point results to `/groups`.

See [docs/STORAGE.md](docs/STORAGE.md) for the full storage strategy.

---

## Syncing Results

```bash
# Check status + download
python3 hpc/scripts/sync_results.py --config hpc/config/myconfig.json

# Just check what's there
python3 hpc/scripts/sync_results.py --config hpc/config/myconfig.json --list-only

# Download but keep on HPC
python3 hpc/scripts/sync_results.py --config hpc/config/myconfig.json --keep-remote
```

By default, files are deleted from HPC after successful transfer to save group storage.

---

## Parameter Sweeps

### 1. Create trial configs in `trials/`
### 2. Generate trial list
```bash
ls trials/*.json | xargs -n1 basename > hpc/scripts/trial_list.txt
```
### 3. Edit and submit
```bash
# Update --array range in run_array.slurm, then:
sbatch hpc/scripts/run_array.slurm
```

### 4. Monitor with `-r` flag
```bash
squeue -r --me   # CRITICAL: -r expands array sub-tasks
```

---

## Monitoring

| Command | Purpose |
|---------|---------|
| `squeue --me` | Your running/pending jobs |
| `squeue -r --job=JOBID` | Array sub-tasks (**always use -r**) |
| `seff JOBID` | CPU/memory efficiency (after completion) |
| `scancel JOBID` | Cancel a job |
| `va` | Remaining CPU hours |
| `uquota` | Storage quotas |

**After every run:** Check `seff JOBID`. If efficiency < 80%, reduce resource requests.

---

## Quick Reference

### Hosts

| Purpose | Hostname |
|---------|----------|
| SSH login | `hpc.arizona.edu` → type `shell` |
| File transfer | `filexfer.hpc.arizona.edu` |
| Web portal | `ood.hpc.arizona.edu` |

### Puma specs

94 CPUs/node, 512 GB RAM, 5 GB/CPU, max 240h walltime, max 1000 simultaneous jobs

See [docs/CHEATSHEET.md](docs/CHEATSHEET.md) for a printable reference card.

---

## Common Gotchas

| Problem | Fix |
|---------|-----|
| Can't login | `/home` full → delete files. Or 3 failed passwords → wait 1h |
| Job killed immediately | Don't compute on login nodes. Use `sbatch` or `interactive` |
| `matplotlib` crash | Set `export MPLBACKEND=Agg` in SLURM scripts |
| Array job shows 1 line | Use `squeue -r` (always!) |
| Transfer fails | Use `filexfer`, not the bastion (10 MB limit) |
| Import errors | `module load python/3.11/3.11.4` + `source activate` first |
| `/xdisk` data gone | Expires after 300 days. Sync locally first |

---

## Publication Acknowledgment

> "This material is based upon High Performance Computing (HPC) resources supported by the University of Arizona TRIF, UITS, and Research, Innovation, and Impact (RII) and maintained by the UArizona Research Technologies department."
