#!/bin/bash
#SBATCH -J simulation
#SBATCH -p genoa
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH -t 72:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=<your-email>

# Generic SLURM job for the HPC simulation skeleton.
# Contract with the R script:
#   * $HOME/${base_name}.R exists and follows hpc_simulation_skeleton.R
#   * it receives the core count as its single command-line argument
#   * it writes $TMPDIR/${base_name}_output.rds
# Change base_name (and num_cores / partition) — nothing else.

# Load necessary modules
module purge
module load 2024
module load R/4.4.2-gfbf-2024a

# Personal R library
export R_LIBS=$HOME/rpackages

export CXX11="g++ -std=gnu++17"
export CXX14="g++ -std=gnu++17"
export CXX17="g++ -std=gnu++17"

# One thread per worker: registerDoParallel() forks one R process per core;
# without this BLAS/LAPACK and OpenMP each start their own thread pool in
# every fork (cores x cores threads), which runs slower than a single core
# and can deadlock across a fork.
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1

# Define base filename (change this only once)
base_name="..."

# Number of cores on the node (genoa: 192)
num_cores=192

echo "TMPDIR is set to: $TMPDIR"
echo "HOME is set to: $HOME"

# Copy R script to scratch directory
r_script="$HOME/${base_name}.R"
if [ -f "$r_script" ]; then
    cp "$r_script" "$TMPDIR"
else
    echo "Error: R script not found at $r_script"
    exit 1
fi

# Optional helpers, only present for some simulations.
for helper in evaluation_functions.R dgp.R; do
    if [ -f "$HOME/$helper" ]; then
        cp "$HOME/$helper" "$TMPDIR"
    fi
done

cd "$TMPDIR"

# Run the R script
Rscript "${base_name}.R" $num_cores
if [ $? -ne 0 ]; then
    echo "Error: R script execution failed."
    exit 1
fi

# Copy the results back
output_file="${base_name}_output.rds"
if [ -f "$TMPDIR/$output_file" ]; then
    mkdir -p "$HOME/sim"
    cp "$TMPDIR/$output_file" "$HOME/sim"
    echo "Results successfully copied to $HOME/sim/$output_file"
else
    echo "Error: Output file not created."
    ls -l $TMPDIR
    exit 1
fi

# Some simulations also write a summary table and a record of the settings
# that produced it; copy those back too if present.
for ext in summary.csv summary.txt; do
    if [ -f "$TMPDIR/${base_name}_${ext}" ]; then
        cp "$TMPDIR/${base_name}_${ext}" "$HOME/sim"
        echo "Copied ${base_name}_${ext} to $HOME/sim"
    fi
done
