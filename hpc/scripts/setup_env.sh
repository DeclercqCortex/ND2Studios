#!/bin/bash
# ============================================================================
# One-Time HPC Python Environment Setup
# ============================================================================
# Run ONCE on a compute node (NOT a login node):
#
#   interactive -a YOUR_GROUP -n 4 -t 1:00:00
#   bash hpc/scripts/setup_env.sh
# ============================================================================

set -euo pipefail

# --- Configuration (edit these) ---
PYTHON_MODULE="python/3.11/3.11.4"
VENV_PATH="$HOME/my-project-env"
# For tight /home quotas, use /groups instead:
# VENV_PATH="/groups/YOUR_GROUP/envs/my-project-env"

# --- Setup ---
echo "=== Setting up Python environment on $(hostname) ==="
echo "Python module: $PYTHON_MODULE"
echo "Venv path: $VENV_PATH"

module load "$PYTHON_MODULE"

if [ -d "$VENV_PATH" ]; then
    echo "WARNING: $VENV_PATH already exists. Skipping creation."
    echo "Delete it first if you want a fresh environment."
else
    python3 -m venv --system-site-packages "$VENV_PATH"
    echo "Created virtual environment at $VENV_PATH"
fi

source "$VENV_PATH/bin/activate"

pip install --upgrade pip
pip install numpy scipy matplotlib

# Add your project's dependencies here:
# pip install pandas scikit-learn torch
# pip install -r requirements.txt

echo ""
echo "=== Environment ready ==="
echo "To activate in future sessions or SLURM scripts:"
echo "  module load $PYTHON_MODULE"
echo "  source $VENV_PATH/bin/activate"
