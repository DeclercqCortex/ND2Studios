# UA HPC Cheatsheet

## Connect
```bash
ssh uahpc                                    # → type 'shell' → Puma
interactive -a GROUP -n 4 -t 1:00:00        # compute session
```

## Submit
```bash
sbatch script.slurm                          # single job
sbatch --array=1-100%10 script.slurm         # array (max 10 concurrent)
```

## Monitor
```bash
squeue --me                 # your jobs
squeue -r --job=JOBID       # array sub-tasks (ALWAYS -r)
seff JOBID                  # efficiency (after completion)
scancel JOBID               # cancel
va                          # CPU hours remaining
uquota                      # storage usage
```

## Transfer (always filexfer!)
```bash
rsync -ravz --exclude .git ./ uahpc-xfer:~/project/              # upload
rsync -ravz uahpc-xfer:/groups/GROUP/results/ ./results/          # download
rsync -ravz --remove-source-files uahpc-xfer:/groups/... ./...    # download + cleanup
```

## Python
```bash
module load python/3.11/3.11.4
source ~/my-env/bin/activate
export MPLBACKEND=Agg
```

## Storage
| `/home` 50 GB | Code only |
| `/groups` 500 GB | Results, envs |
| `/xdisk` ≤20 TB | Temp (300-day limit) |

## SLURM Directives
```bash
#SBATCH --account=GROUP     --partition=standard
#SBATCH --nodes=1           --ntasks=1
#SBATCH --cpus-per-task=4   # × 5 GB RAM
#SBATCH --time=04:00:00     # max 240:00:00
#SBATCH --array=1-100%10    # 100 tasks, max 10 concurrent
```

## Don't Forget
- `squeue -r` for arrays
- `filexfer` for transfers (bastion = 10 MB limit)
- `MPLBACKEND=Agg` in SLURM
- Never compute on login nodes
- `seff` after every run
- Sync locally → delete from HPC
