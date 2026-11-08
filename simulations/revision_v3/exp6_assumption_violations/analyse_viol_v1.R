# ══════════════════════════════════════════════════════════════════════════════
# analyse_viol_v1.R -- local analysis of experiment 6 (violations of the
# assumptions).
#
# Reads results/sim_viol_v1_output.rds (one row per cell x replication x method
# x target x population, produced on the HPC by sim_viol_v1.R) and writes, to
# results/tables/:
#   summary.csv          mean metrics per (cell, method, target, population)
#   censoring.csv        realised censoring fractions per cell
#   viol_table.tex       CATE metrics over the combined population, all scenarios
#   viol_table_detail.tex  transportability and overlap, by target and population
# and prints the main table.  The .tex files are pasted into the supplement by
# hand.  Base R only.  Run from the repository root:
#   Rscript simulations/revision_v3/exp6_assumption_violations/analyse_viol_v1.R
# ══════════════════════════════════════════════════════════════════════════════

res_dir <- "simulations/revision_v3/exp6_assumption_violations/results"
tab_dir <- file.path(res_dir, "tables")
dir.create(tab_dir, recursive = TRUE, showWarnings = FALSE)

res <- readRDS(file.path(res_dir, "sim_viol_v1_output.rds"))
# The one-sided overlap scenario (trial x1 <= 0) is not reported: the trial-only
# comparator is biased already inside its support there, for reasons we have
# not traced, and the two-sided scenario makes the overlap point on its own.
res <- res[res$cell != "overlap_one_sided", ]
cell_names <- unique(res$cell)                 # in the order of sim_viol_v1.R
cat(sprintf("Loaded %d rows; %d cells; %d reps per cell\n", nrow(res),
            length(cell_names), max(res$Iter)))

meth <- c("Fusion", "RCT-only", "RWD-only")

# ── Summary table ─────────────────────────────────────────────────────────────
grp <- cbind(rmse, bias, coverage, width, postvar) ~
  cell + method + target + population
summary_tbl <- aggregate(grp, data = res,
                         FUN = function(x) mean(x, na.rm = TRUE),
                         na.action = na.pass)
n_tbl <- aggregate(rmse ~ cell + method + target + population, data = res,
                   FUN = function(x) sum(is.finite(x)), na.action = na.pass)
names(n_tbl)[ncol(n_tbl)] <- "n"
summary_tbl <- merge(summary_tbl, n_tbl)
summary_tbl$cell <- factor(summary_tbl$cell, levels = cell_names)
summary_tbl <- summary_tbl[order(summary_tbl$cell, summary_tbl$target,
                                 summary_tbl$population, summary_tbl$method), ]
write.csv(summary_tbl, file.path(tab_dir, "summary.csv"), row.names = FALSE)

cat("\n=== Fits per cell and method (target tau1, population All) ===\n")
print(with(summary_tbl[summary_tbl$target == "tau1" &
                         summary_tbl$population == "All", ],
           tapply(n, list(cell, method), sum)))

cat("\n=== CATE metrics, target tau1, population All ===\n")
print(summary_tbl[summary_tbl$target == "tau1" &
                    summary_tbl$population == "All",
                  c("cell", "method", "n", "rmse", "bias", "coverage", "width",
                    "postvar")], row.names = FALSE, digits = 3)

# Realised censoring fractions per cell (sanity check).
one <- res[res$method == "Fusion" & res$target == "tau1" &
             res$population == "All", ]
cens_tbl <- aggregate(cbind(cens_rct, ic_rwd, rc_rwd_tail) ~ cell, data = one,
                      FUN = function(x) mean(x, na.rm = TRUE))
cens_tbl$cell <- factor(cens_tbl$cell, levels = cell_names)
cens_tbl <- cens_tbl[order(cens_tbl$cell), ]
write.csv(cens_tbl, file.path(tab_dir, "censoring.csv"), row.names = FALSE)
cat("\n=== Realised censoring fractions ===\n")
print(cens_tbl, row.names = FALSE, digits = 3)

# ── LaTeX helpers ─────────────────────────────────────────────────────────────
num <- function(x, d = 2) {
  if (length(x) == 0 || is.na(x)) return("---")
  s <- formatC(x, format = "f", digits = d)
  if (x < 0) paste0("$", s, "$") else s
}
get_val <- function(cl, m, tg, pop, col) {
  summary_tbl[[col]][summary_tbl$cell == cl & summary_tbl$method == m &
                       summary_tbl$target == tg &
                       summary_tbl$population == pop]
}
triple <- function(cl, m, tg, pop) {
  paste(vapply(c("rmse", "bias", "coverage"),
               function(k) num(get_val(cl, m, tg, pop, k)), ""),
        collapse = " & ")
}

blocks <- list(
  list(title = NULL, cells = c(reference = "No violation")),
  list(title = "Error law (Assumption~6)",
       cells = c(logistic_lu1    = "Logistic selection, $\\lambda_u = 1$",
                 logistic_lu2    = "Logistic selection, $\\lambda_u = 2$",
                 "trt_scale_1.5" = "Treated-arm scale $\\times 1.5$",
                 trt_scale_2     = "Treated-arm scale $\\times 2$",
                 x_scale         = "Covariate-dependent scale")),
  list(title = "Short trial follow-up",
       cells = c(short_followup        = "No violation",
                 "short_trt_scale_1.5" = "Treated-arm scale $\\times 1.5$")),
  list(title = "Transportability (Assumption~4)",
       cells = c("transport_0.5" = "$\\kappa = 0.5$",
                 transport_1     = "$\\kappa = 1$")),
  list(title = "Overlap (Assumption~3)",
       cells = c(overlap_two_sided = "Trial $|x_1| \\leq 1$")),
  list(title = "Informative censoring (Assumption~5)",
       cells = c(dropout_null         = "Non-informative dropout",
                 dropout_informative  = "Informative, both arms",
                 dropout_differential = "Informative, treated arm"))
)

# ── Table: CATE metrics over the combined population, all scenarios ───────────
rows <- character(0)
for (b in blocks) {
  if (!is.null(b$title)) {
    rows <- c(rows, "    \\addlinespace",
              sprintf("    \\multicolumn{10}{l}{\\emph{%s}}\\\\", b$title))
  }
  for (cl in names(b$cells)) {
    rows <- c(rows, sprintf("    %s%s & %s \\\\",
                            if (is.null(b$title)) "" else "\\quad ", b$cells[[cl]],
                            paste(vapply(meth, function(m)
                              triple(cl, m, "tau1", "All"), ""),
                              collapse = " & ")))
  }
}
writeLines(c(
  "% Generated by simulations/revision_v3/exp6_assumption_violations/analyse_viol_v1.R -- do not edit.",
  "\\begin{table}[!htbp]",
  "  \\centering",
  "  \\color{revisioncol}",
  "  \\caption{CATE metrics when an assumption fails, for the Bayesian fusion",
  "    forest and the trial-only and real-world-only causal forests. Each scenario",
  "    changes one feature of the design of the main text at",
  "    $\\lambda_d = \\lambda_u = 1$. Estimates are scored against the trial CATE",
  "    $\\tau(x)$ over the combined population, and coverage refers to $95\\%$",
  sprintf("    credible intervals. Averages over $%d$ replications.}", max(summary_tbl$n)),
  "  \\label{tab:sim-viol}",
  "  \\small",
  "  \\begin{tabular}{@{}l rrr rrr rrr@{}}",
  "    \\toprule",
  "    & \\multicolumn{3}{c}{Fusion forest} & \\multicolumn{3}{c}{Trial only} & \\multicolumn{3}{c}{Real-world only} \\\\",
  "    \\cmidrule(lr){2-4} \\cmidrule(lr){5-7} \\cmidrule(l){8-10}",
  "    Scenario & RMSE & Bias & Cov. & RMSE & Bias & Cov. & RMSE & Bias & Cov. \\\\",
  "    \\midrule",
  rows,
  "    \\bottomrule",
  "  \\end{tabular}",
  "\\end{table}"), file.path(tab_dir, "viol_table.tex"))

# ── Table: transportability and overlap, by target and population ─────────────
det <- list(
  list(title = "Transportability, $\\kappa = 0.5$", cell = "transport_0.5"),
  list(title = "Transportability, $\\kappa = 1$", cell = "transport_1"))
rows <- character(0)
for (b in det) {
  rows <- c(rows, sprintf("    \\multicolumn{5}{l}{\\emph{%s}}\\\\", b$title),
            sprintf("    \\quad $\\widehat\\tau$ against $\\tau$, trial rows & %s & %s \\\\",
                    triple(b$cell, "Fusion", "tau1", "RCT"),
                    num(get_val(b$cell, "Fusion", "tau1", "RCT", "width"))),
            sprintf("    \\quad $\\widehat\\tau$ against $\\tau + \\eta$, real-world rows & %s & %s \\\\",
                    triple(b$cell, "Fusion", "tau0", "RWD"),
                    num(get_val(b$cell, "Fusion", "tau0", "RWD", "width"))),
            sprintf("    \\quad $\\widehat c$ against $\\eta + \\lambda_u \\delta$, real-world rows & %s & %s \\\\",
                    triple(b$cell, "Fusion", "c", "RWD"),
                    num(get_val(b$cell, "Fusion", "c", "RWD", "width"))),
            "    \\addlinespace")
}
ovl <- list(
  list(title = "Overlap, trial $|x_1| \\leq 1$", cell = "overlap_two_sided"))
for (b in ovl) {
  rows <- c(rows, sprintf("    \\multicolumn{5}{l}{\\emph{%s}}\\\\", b$title))
  for (m in c("Fusion", "RCT-only")) for (p in c("RWD-in", "RWD-out")) {
    rows <- c(rows, sprintf("    \\quad %s, real-world rows %s the trial support & %s & %s \\\\",
                            if (m == "Fusion") "Fusion forest" else "Trial only",
                            if (p == "RWD-in") "inside" else "outside",
                            triple(b$cell, m, "tau1", p),
                            num(get_val(b$cell, m, "tau1", p, "width"))))
  }
  if (!identical(b, ovl[[length(ovl)]])) rows <- c(rows, "    \\addlinespace")
}
writeLines(c(
  "% Generated by simulations/revision_v3/exp6_assumption_violations/analyse_viol_v1.R -- do not edit.",
  "\\begin{table}[!htbp]",
  "  \\centering",
  "  \\color{revisioncol}",
  "  \\caption{The transportability and overlap scenarios of",
  "    Web Table~\\ref{tab:sim-viol} by target and population. Coverage and width",
  sprintf("    refer to $95\\%%$ credible intervals. Averages over $%d$ replications.}", max(summary_tbl$n)),
  "  \\label{tab:sim-viol-detail}",
  "  \\small",
  "  \\begin{tabular}{@{}l rrrr@{}}",
  "    \\toprule",
  "    Estimate and target & RMSE & Bias & Coverage & Width \\\\",
  "    \\midrule",
  rows,
  "    \\bottomrule",
  "  \\end{tabular}",
  "\\end{table}"), file.path(tab_dir, "viol_table_detail.tex"))

cat("\nWrote", file.path(tab_dir, "summary.csv"), ",",
    file.path(tab_dir, "censoring.csv"), ",",
    file.path(tab_dir, "viol_table.tex"), "and",
    file.path(tab_dir, "viol_table_detail.tex"), "\n")
