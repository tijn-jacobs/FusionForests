# Simulations

Four experiments behind the simulation study (`notes/simulation/SIMULATION.tex`).
Each folder holds its R script(s), `*_output.rds`, and `figures/`.

| Folder | Experiment | Script(s) |
|--------|------------|-----------|
| `exp1_confounding_heterogeneity/` | Confounding & between-source heterogeneity sweep | `sim_surv_v12.R` |
| `exp2_competitor/` | Deep-learning (deepAFT) competitor | `sim_surv_v12_{bff,deepaft,rsf}.R`, `plots_bff_dnn.R`, `tables_bff_dnn.R` |
| `exp3_covariate_dimension/` | Growing covariate dimension `p` | `sim_hd_v1{a,b,c,d}.R` → `sim_hd_combine.R` |
| `exp4_error_distribution/` | Nonparametric (HDPM) error distribution | `sim_err_v2.R` |

Run scripts from the repository root. The manuscript's copies of the figures
live in `notes/general/figures/`; the `figures/` here are the scripts' working
output (copy over when updating the paper).
