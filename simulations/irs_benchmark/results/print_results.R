# ============================================================================
# IRS Benchmark — Preliminary Results Summary
# ============================================================================
#
# Reads preliminary_results_full.csv and prints formatted tables.
# Shows RMSE (train), Coverage (train), and RMSE (test) per method.
# Usage: Rscript print_results.R
# ============================================================================

script_dir <- dirname(sub("--file=", "",
  commandArgs(trailingOnly = FALSE)[
    grep("--file=", commandArgs(trailingOnly = FALSE))]))
if (length(script_dir) == 0 || script_dir == "") script_dir <- "."

res <- read.csv(file.path("/Users/tijnjacobs/Library/CloudStorage/OneDrive-VrijeUniversiteitAmsterdam/Documents/GitHub/FusionForests/simulations/irs_benchmark/results/preliminary_results_full.csv"),
                check.names = FALSE)

methods_order <- c("Oracle", "IRS (informed)", "IRS (uniform)",
                   "Complete case", "bartMachine MIA",
                   "missForest+BART")
method_labels <- c(
  "Oracle"          = "Oracle",
  "IRS (informed)"  = "IRS-I",
  "IRS (uniform)"   = "IRS-U",
  "Complete case"   = "CC",
  "bartMachine MIA" = "bartM",
  "missForest+BART" = "mfBART"
)

# --- Header -----------------------------------------------------------------

cat(strrep("=", 78), "\n")
cat("  IRS BENCHMARK — PRELIMINARY RESULTS\n")
cat(strrep("=", 78), "\n\n")

cat("SIMULATION DESIGN\n")
cat(strrep("-", 78), "\n")
cat("
  Data:       y = f(X) + eps,  eps ~ N(0, 0.5^2),  d = 10 covariates
  Test set:   n_test = 1000 (fixed)
  BART:       200 trees, 3000 posterior / 2000 burn-in
  Reps:       191 per scenario (main), 192 (bartMachine)
  Missingness imposed on columns 1-3 only

  Regression functions:
    quadratic  :  X1^2 + X2^2 + X3^2
    linear     :  beta'X,  beta = (1, 2, -1, 3, -0.5, -1, 0.3, 1.7, 0.4, -0.3)
    friedman   :  10*sin(pi*X1*X2) + 20*(X3-0.5)^2 + 10*X4 + 5*X5
    nonlinear  :  sin(pi*X1*X2) + 2*(X3-0.5)^2 + X4 + 0.5*X5
                  (covariates are nonlinear transforms of a hidden variable)

  Missingness mechanisms:
    MCAR        :  independent Bernoulli(p) per entry
    MNAR        :  values above (1-p) quantile are masked
    Predictive  :  MCAR pattern, but y regenerated as sum(Xj^2 + 2*Mj) + eps

  Methods:
    Oracle  = BART on complete data (lower bound on RMSE)
    IRS-I   = SimpleBART with informed random splitting (irs=2)
    IRS-U   = SimpleBART with uniform random splitting (irs=3)
    CC      = Complete-case analysis (drop rows with NaN)
    bartM   = bartMachine with missing-in-attributes (MIA)
    mfBART  = missForest imputation + standard BART

  Status: INCOMPLETE — main sim: 48/108 scenarios; bartM: 12/108 scenarios
")
cat(strrep("-", 78), "\n\n")

# --- Helper functions -------------------------------------------------------

fmt_val <- function(x) {
  ifelse(is.na(x), "    --", sprintf("%6.2f", x))
}

fmt_cov <- function(x) {
  ifelse(is.na(x), "    --", sprintf("%5.1f%%", x * 100))
}

# Pivot long data to wide for a given metric
pivot_metric <- function(df, metric) {
  wide <- reshape(df[, c("model", "pattern", "p_miss", "n_train",
                          "method", metric)],
                  idvar = c("model", "pattern", "p_miss", "n_train"),
                  timevar = "method",
                  direction = "wide")
  # Clean column names
  names(wide) <- gsub(paste0(metric, "\\."), "", names(wide))
  wide
}

# --- Print tables by pattern ------------------------------------------------

for (pat in c("mcar", "mnar", "predictive")) {
  pat_label <- switch(pat,
    mcar = "MCAR (Missing Completely at Random)",
    mnar = "MNAR (Missing Not at Random)",
    predictive = "Predictive Missingness"
  )

  cat(strrep("=", 78), "\n")
  cat(sprintf("  %s\n", pat_label))
  cat(strrep("=", 78), "\n")

  sub <- res[res$pattern == pat, ]

  for (nt in sort(unique(sub$n_train))) {
    for (pm in sort(unique(sub[sub$n_train == nt, "p_miss"]))) {

      cat(sprintf("\n  n_train = %d,  p_miss = %.2f\n", nt, pm))
      cat(sprintf("  %s\n", strrep("-", 74)))

      ss <- sub[sub$n_train == nt & sub$p_miss == pm, ]

      # Header
      cat(sprintf("  %-12s  %-8s  %6s %6s %6s %6s %6s %6s\n",
                  "model", "metric",
                  "Oracle", "IRS-I", "IRS-U", "CC", "bartM", "mfBART"))
      cat(sprintf("  %s\n", strrep("-", 74)))

      for (mod in c("quadratic", "linear", "friedman", "nonlinear")) {
        rows <- ss[ss$model == mod, ]
        if (nrow(rows) == 0) next

        # Get values for each method
        get_val <- function(method_name, col) {
          r <- rows[rows$method == method_name, ]
          if (nrow(r) == 0) return(NA)
          r[[col]]
        }

        meths <- methods_order

        # RMSE train
        v <- sapply(meths, function(m) get_val(m, "rmse_train"))
        cat(sprintf("  %-12s  %-8s  %6s %6s %6s %6s %6s %6s\n",
                    mod, "RMSE-tr",
                    fmt_val(v[1]), fmt_val(v[2]), fmt_val(v[3]),
                    fmt_val(v[4]), fmt_val(v[5]), fmt_val(v[6])))

        # Coverage train
        v <- sapply(meths, function(m) get_val(m, "coverage"))
        cat(sprintf("  %-12s  %-8s  %6s %6s %6s %6s %6s %6s\n",
                    "", "Cov-tr",
                    fmt_cov(v[1]), fmt_cov(v[2]), fmt_cov(v[3]),
                    fmt_cov(v[4]), fmt_cov(v[5]), fmt_cov(v[6])))

        # RMSE test
        v <- sapply(meths, function(m) get_val(m, "rmse_test"))
        cat(sprintf("  %-12s  %-8s  %6s %6s %6s %6s %6s %6s\n",
                    "", "RMSE-te",
                    fmt_val(v[1]), fmt_val(v[2]), fmt_val(v[3]),
                    fmt_val(v[4]), fmt_val(v[5]), fmt_val(v[6])))

        cat("\n")
      }
    }
  }
}

# --- Conclusions ------------------------------------------------------------

cat(strrep("=", 78), "\n")
cat("  PRELIMINARY CONCLUSIONS\n")
cat(strrep("=", 78), "\n")
cat("
  1. IRS-informed and IRS-uniform perform nearly identically across all
     settings — the informed split prior adds negligible benefit.

  2. Under MCAR, imputation-based methods (missForest+BART, bartMachine
     MIA) tend to outperform IRS, especially for standard covariate
     structures (quadratic, linear, friedman). Complete-case analysis
     is competitive at low missingness (p=0.25).

  3. Under MNAR, IRS clearly dominates. Complete-case analysis suffers
     severe bias (systematically drops large values). Imputation methods
     (missForest, bartMachine) also degrade because they cannot recover
     the truncated distribution. IRS handles this gracefully by
     splitting around the missing values without imputing.
     The advantage is most dramatic on the nonlinear DGP (manifold
     structure): IRS RMSE ~0.6 vs CC ~6.0 and bartM ~1.5.

  4. Under predictive missingness, all methods (including the Oracle)
     show similar test RMSE for friedman and nonlinear DGPs, because
     the test set has no missingness (M=0) — the predictive signal
     from M is absent at test time.

  5. These results are preliminary (48/108 main scenarios, 12/108
     bartMachine scenarios). Larger sample sizes (n=400) and the
     remaining bartMachine comparisons are still running.
")
