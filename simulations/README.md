# Simulations

Two rounds, one folder each, with the same internal layout (one folder per
experiment of the paper's simulation study):

| Folder | Round |
|--------|-------|
| `submission_v2/` | The simulations behind the submitted manuscript (`manuscripts/manuscript_v2`). Frozen. |
| `revision_v3/`   | The simulations for the revision (`manuscripts/manuscript_v3`). See its README for what changed. |
| `benchmark_centering/` | Sampler before/after a fix, paired on identical data: the centring fix (package 1.1.0 vs 1.1.1) and the per-source augmentation scale (1.1.1 vs 1.1.2). |

`hpc_simulation_skeleton.R` + `hpc_job_skeleton.sh` are reusable templates for
new HPC simulation scripts: they document the shared section layout and the
`<base_name>.R` → `<base_name>_output.rds` naming contract that the SLURM job
script relies on. Run scripts from the repository root.
