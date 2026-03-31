#!/bin/bash
#
# Submit all IRS benchmark v2 scenarios as a SLURM job array.
#
# Usage:
#   sbatch simulations/irs_benchmark_v2/submit_all.sh
#
# 24 scenarios: 4 outcomes x 3 missingness x 2 rho
#
#SBATCH --job-name=irs_v2
#SBATCH --array=1-24
#SBATCH --cpus-per-task=16
#SBATCH --mem=16G
#SBATCH --time=08:00:00
#SBATCH --output=logs/irs_v2_%a.out
#SBATCH --error=logs/irs_v2_%a.err

module load R

mkdir -p logs

Rscript simulations/irs_benchmark_v2/run_simulation.R \
  ${SLURM_CPUS_PER_TASK} \
  ${SLURM_ARRAY_TASK_ID}
