# ============================================================================
# Scenario grid for IRS benchmark v2
#
# 4 outcome scenarios x 3 missingness patterns x 2 rho values = 24 configs
#
# Use get_scenario(id) to retrieve a single scenario by integer ID.
# ============================================================================

build_scenario_grid <- function() {
  grid <- expand.grid(
    outcome_scenario = 0:3,
    miss_pattern     = c("block_rct", "block_rwd", "mcar"),
    rho              = c(0, 0.5),
    stringsAsFactors = FALSE
  )
  grid$scenario_id <- seq_len(nrow(grid))

  # Label for outcome scenario
  grid$outcome_label <- c(
    "irrelevant", "prognostic",
    "effect_modifier", "prognostic_em"
  )[grid$outcome_scenario + 1]

  grid
}

get_scenario <- function(id) {
  grid <- build_scenario_grid()
  if (id < 1 || id > nrow(grid)) {
    stop(sprintf("Scenario ID %d out of range [1, %d]",
                 id, nrow(grid)))
  }
  as.list(grid[id, ])
}

get_n_scenarios <- function() {
  nrow(build_scenario_grid())
}

print_scenario_grid <- function() {
  grid <- build_scenario_grid()
  cat(sprintf("Total scenarios: %d\n\n", nrow(grid)))
  print(grid[, c("scenario_id", "outcome_scenario",
                  "outcome_label", "miss_pattern", "rho")])
}
