# ============================================================================
# Scenario grid for the IRS benchmark simulation
#
# Each scenario is one row of the grid:
#   model x pattern x p_miss x n_train x d
#
# Use get_scenario(id) to retrieve a single scenario by integer ID.
# This allows the HPC job array to dispatch one scenario per job.
# ============================================================================

build_scenario_grid <- function() {
  grid <- expand.grid(
    model   = c("quadratic", "linear", "friedman", "nonlinear"),
    pattern = c("mcar", "mnar", "predictive"),
    p_miss  = c(0.1, 0.3, 0.5, 0.7),
    n_train = c(100, 200, 500, 1000),
    stringsAsFactors = FALSE
  )

  # d is fixed at 10 for all models (model 4 ignores it)
  grid$d <- 10L

  # miss_cols: columns 1:3 for all scenarios
  grid$miss_cols <- replicate(nrow(grid), 1:3, simplify = FALSE)

  # Predictive pattern only makes sense with model = "quadratic"
  # (since Y = sum X_j^2 + 2*M_j uses the same functional form)
  # Keep all combinations but flag this
  grid$scenario_id <- seq_len(nrow(grid))

  grid
}

get_scenario <- function(id) {
  grid <- build_scenario_grid()
  if (id < 1 || id > nrow(grid)) {
    stop(sprintf("Scenario ID %d out of range [1, %d]", id, nrow(grid)))
  }
  as.list(grid[id, ])
}

get_n_scenarios <- function() {
  nrow(build_scenario_grid())
}

# Print the full grid for inspection
print_scenario_grid <- function() {
  grid <- build_scenario_grid()
  cat(sprintf("Total scenarios: %d\n\n", nrow(grid)))
  print(grid[, c("scenario_id", "model", "pattern", "p_miss", "n_train", "d")])
}
