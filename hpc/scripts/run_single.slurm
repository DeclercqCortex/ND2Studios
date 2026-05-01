#!/bin/bash
# ============================================================================
# Single Job SLURM Template
# ============================================================================
# Submit: sbatch hpc/scripts/run_single.slurm
# On Puma, each CPU = 5 GB RAM (4 CPUs = 20 GB).
# ============================================================================

#SBATCH --job-name=my-job
#SBATCH --account=YOUR_GROUP
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=04:00:00
#SBATCH --output=slurm_logs/%x_%j.out
#SBATCH --error=slurm_logs/%x_%j.err

# --- Environment ---
module load python/3.11/3.11.4
source ~/my-project-env/bin/activate
export MPLBACKEND=Agg

mkdir -p slurm_logs
cd ~/my-project

# --- Run ---
echo "=== Job $SLURM_JOB_ID on $(hostname) at $(date) ==="
echo "CPUs: $SLURM_CPUS_ON_NODE"

# Replace with your command:
python3 src/main.py --output-dir /groups/YOUR_GROUP/results/run_$SLURM_JOB_ID

echo "=== Done at $(date). Check: seff $SLURM_JOB_ID ==="
