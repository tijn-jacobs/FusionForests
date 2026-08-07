# Simulations

`hpc_simulation_skeleton.R` + `hpc_job_skeleton.sh` are reusable templates for
new HPC simulation scripts (also for other projects): they document the shared
section layout and the `<base_name>.R` → `<base_name>_output.rds` naming
contract that the SLURM job script relies on.

Four experiments behind the simulation study (`notes/simulation/SIMULATION.tex`).
Each folder holds its R script(s), `*_output.rds`, and `figures/`.

| Folder | Experiment | Script(s) |
|--------|------------|-----------|
| `exp1_confounding_heterogeneity/` | Confounding & between-source heterogeneity sweep | `sim_surv_v12.R` |
| `exp2_competitor/` | Deep-learning (deepAFT) competitor | `sim_surv_v12_{bff,deepaft}.R` → `present_results.R`, `plots_bff_dnn.R`, `tables_bff_dnn.R` |
| `exp3_covariate_dimension/` | Growing covariate dimension `p` | `sim_hd_v1{a,b,c,d}.R` → `sim_hd_combine.R` |
| `exp4_error_distribution/` | Nonparametric (HDPM) error distribution | `sim_err_v2.R` |

Run scripts from the repository root. The figure scripts
(`sim_surv_v12.R`, `plots_bff_dnn.R`, `sim_hd_combine.R`) write their
manuscript figures straight to `notes/general/figures/`, so a re-run updates
the paper directly. Any `figures/` folders left inside the `exp*` directories
are stale working output from before this change and can be ignored.
