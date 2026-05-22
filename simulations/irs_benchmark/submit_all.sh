#!/bin/bash
#
# Submit all IRS benchmark scenarios as a SLURM job array.
#
# Usage:
#   sbatch simulations/irs_benchmark/submit_all.sh
#
# Each array task runs one scenario (model x pattern x p_miss x n_train).
# Use `Rscript simulations/irs_benchmark/scenarios.R` to inspect the grid.
#
#SBATCH --job-name=irs_benchmark
#SBATCH --array=1-192
#SBATCH --cpus-per-task=16
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --output=logs/irs_benchmark_%a.out
#SBATCH --error=logs/irs_benchmark_%a.err

module load R

mkdir -p logs

Rscript simulations/irs_benchmark/run_simulation.R \
  ${SLURM_CPUS_PER_TASK} \
  ${SLURM_ARRAY_TASK_ID}
