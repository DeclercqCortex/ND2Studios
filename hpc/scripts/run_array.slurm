#!/bin/bash
# ============================================================================
# Array Job SLURM Template (Parameter Sweeps)
# ============================================================================
# Setup:
#   1. Put trial configs in trials/
#   2. ls trials/*.json | xargs -n1 basename > hpc/scripts/trial_list.txt
#   3. Update --array range below
#   4. sbatch hpc/scripts/run_array.slurm
# ============================================================================

#SBATCH --job-name=sweep
#SBATCH --account=YOUR_GROUP
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=04:00:00
#SBATCH --array=1-100%10
#SBATCH --output=slurm_logs/%x_%A_%a.out
#SBATCH --error=slurm_logs/%x_%A_%a.err

# --- Environment ---
module load python/3.11/3.11.4
source ~/my-project-env/bin/activate
export MPLBACKEND=Agg

cd ~/my-project
mkdir -p slurm_logs

# --- Results go to /groups (NOT /home) ---
RESULTS_BASE="/groups/YOUR_GROUP/results/sweep"
mkdir -p "$RESULTS_BASE"

# --- Disk space check ---
AVAIL_KB=$(df --output=avail "$RESULTS_BASE" 2>/dev/null | tail -1)
AVAIL_GB=$(( ${AVAIL_KB:-0} / 1048576 ))
if [ "$AVAIL_GB" -lt 2 ]; then
    echo "ABORT: Only ${AVAIL_GB} GB free. Need >= 2 GB."
    exit 1
fi

# --- Stagger starts (reduce I/O storms) ---
sleep $(( RANDOM % 30 ))

# --- Look up trial config ---
TRIAL=$(sed -n "${SLURM_ARRAY_TASK_ID}p" hpc/scripts/trial_list.txt)
TRIAL_NAME="${TRIAL%.json}"

if [ -z "$TRIAL" ]; then
    echo "ERROR: No trial for task $SLURM_ARRAY_TASK_ID"
    exit 1
fi

echo "=== Task $SLURM_ARRAY_TASK_ID: $TRIAL ==="
echo "Node: $(hostname), Start: $(date), Disk: ${AVAIL_GB} GB free"

# Replace with your command:
python3 src/main.py --config "trials/$TRIAL" --output-dir "$RESULTS_BASE/$TRIAL_NAME"

echo "Task $SLURM_ARRAY_TASK_ID done at $(date)"
