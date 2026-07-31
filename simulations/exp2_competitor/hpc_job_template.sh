#!/bin/bash
#SBATCH -J simulation
#SBATCH -p genoa
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH -t 72:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=tijnjacobs@outlook.com

# Load necessary modules
module purge
module load 2024
module load R/4.4.2-gfbf-2024a

# Set R library path to include your personal library
export R_LIBS=$HOME/rpackages

export CXX11="g++ -std=gnu++17"
export CXX14="g++ -std=gnu++17"
export CXX17="g++ -std=gnu++17"

# ADDED -- one thread per worker. registerDoParallel() forks one R process per
# core, and each fork would otherwise let BLAS/LAPACK and OpenMP start their own
# thread pool: on a 192-core node that is 192 x 192 threads fighting for the
# machine, which runs slower than a single core and can deadlock across a fork.
# xgboost is already pinned with nthread = 1 inside its script; these cover
# everything else -- BJ-ELM calls ginv() per base learner, which is a BLAS SVD.
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1

# Define base filename (change this only once)
# base_name="sim_surv_v12_xgbaft"     # XGBoost-AFT -- fast; tunes first
# base_name="sim_surv_v12_bjelm"      # BJ-ELM      -- slow
# base_name="sim_surv_v12_bjtree"     # BJ-trees    -- slowest; run TIMING first
base_name="..."

# Define number of cores, 192
num_cores=192


# Debug TMPDIR
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

# Optional helper, only present for some simulations.
if [ -f "$HOME/evaluation_functions.R" ]; then
    cp "$HOME/evaluation_functions.R" "$TMPDIR"
fi

# All three simulations are self-contained: the BJ-ELM implementation is inlined
# in its script, so nothing beyond the .R file has to travel to $TMPDIR.

# Change to scratch directory
cd "$TMPDIR"

# Run the R script
Rscript "${base_name}.R" $num_cores
if [ $? -ne 0 ]; then
    echo "Error: R script execution failed."
    exit 1
fi

# Check if the output file exists and copy it back
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

# ADDED -- the simulations also write a summary table and a record of the
# settings that produced it. For XGBoost-AFT the selected hyperparameters live
# only in the .txt, so losing it loses the tuning result.
for ext in summary.csv summary.txt; do
    if [ -f "$TMPDIR/${base_name}_${ext}" ]; then
        cp "$TMPDIR/${base_name}_${ext}" "$HOME/sim"
        echo "Copied ${base_name}_${ext} to $HOME/sim"
    fi
done
