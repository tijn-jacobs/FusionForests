# ──────────────────────────────────────────────────────────────────────────────
# Combine the competitor results into one table.
#
#   Rscript simulations/exp2_competitor/combine_competitors.R
#
# Reads every *_output.rds in this directory, keeps the cell the main text
# reports (lambda_d = lambda_u = 1), and writes:
#
#   competitors_table.csv   one row per method x population
#   competitors_table.tex   the same, as a booktabs table for the supplement
#
# Base R only. Run from the repository root.
# ──────────────────────────────────────────────────────────────────────────────

# Run from the repository root (no setwd; paths below are repo-relative).
res_dir <- "simulations/exp2_competitor"

files <- c(
  "sim_surv_v12_bff_output.rds",
  "sim_surv_v12_deepaft_output.rds",
  "sim_surv_v12_xgbaft_output.rds",
  "sim_surv_v12_bjelm_output.rds",
  "sim_surv_v12_bjtree_output.rds"
)
files <- files[file.exists(file.path(res_dir, files))]
stopifnot(length(files) > 0)

# The replication count differs between methods, so it is carried through and
# reported: a table that hides it invites the reader to compare Monte Carlo
# errors that are not comparable.
raw <- do.call(rbind, lapply(files, function(f) {
  d <- readRDS(file.path(res_dir, f))
  d <- d[d$lambda_d == 1 & d$lambda_u == 1, ]
  keep <- c("method", "population", "rmse", "bias", "coverage", "width",
            "postvar", "Iter")
  d <- d[, keep]
  d$file <- f
  d
}))

cat("Loaded:\n")
for (f in files) {
  d <- readRDS(file.path(res_dir, f))
  cat(sprintf("  %-34s M = %4d, methods: %s\n", f, max(d$Iter),
              paste(unique(d$method), collapse = ", ")))
}

# ── Aggregate over replications ───────────────────────────────────────────────
agg <- function(v, f) tapply(v, list(raw$method, raw$population), f)
mean_na <- function(x) mean(x, na.rm = TRUE)

methods <- sort(unique(raw$method))
pops    <- c("RCT", "RWD", "All")

tbl <- do.call(rbind, lapply(methods, function(m) {
  do.call(rbind, lapply(pops, function(p) {
    s <- raw[raw$method == m & raw$population == p, ]
    if (!nrow(s)) return(NULL)
    data.frame(
      method     = m,
      population = p,
      M          = max(s$Iter),
      rmse       = mean_na(s$rmse),
      bias       = mean_na(s$bias),
      bias_var   = var(s$bias, na.rm = TRUE),
      coverage   = mean_na(s$coverage),
      width      = mean_na(s$width),
      postvar    = mean_na(s$postvar),
      stringsAsFactors = FALSE)
  }))
}))
tbl$sd <- sqrt(tbl$bias_var)

# ── Ordering: fusion first, then the competitor families ──────────────────────
fam <- function(m) {
  if (m == "Fusion") return(1L)
  if (grepl("^deepAFT", m)) return(2L)
  if (grepl("^xgbAFT",  m)) return(3L)
  if (grepl("^bjelm",   m)) return(4L)
  if (grepl("^bjtree",  m)) return(5L)
  6L
}
# Within a family: RCT-S, RCT-T, Pooled-S, Pooled-T.
variant <- function(m) {
  v <- c("RCT-S" = 1L, "RCT-T" = 2L, "Pooled-S" = 3L, "Pooled-T" = 4L)
  hit <- v[vapply(names(v), function(k) grepl(k, m, fixed = TRUE), logical(1))]
  if (length(hit)) hit[1] else 0L
}
tbl$.fam <- vapply(tbl$method, fam, integer(1))
tbl$.var <- vapply(tbl$method, variant, integer(1))
tbl$population <- factor(tbl$population, levels = pops)
tbl <- tbl[order(tbl$population, tbl$.fam, tbl$.var), ]
tbl$.fam <- tbl$.var <- NULL

# ── Console ───────────────────────────────────────────────────────────────────
cat("\n=== CATE metrics at lambda_d = lambda_u = 1 ===\n")
for (p in pops) {
  s <- tbl[tbl$population == p, ]
  if (!nrow(s)) next
  cat("\npopulation:", p, "\n")
  print(s[, c("method", "M", "rmse", "bias", "sd", "coverage", "width",
              "postvar")],
        row.names = FALSE, digits = 3)
}

# ── CSV ───────────────────────────────────────────────────────────────────────
csv <- file.path(res_dir, "competitors_table.csv")
write.csv(tbl, csv, row.names = FALSE)

# ── LaTeX, for the supplement ─────────────────────────────────────────────────
# Layout: the method family gets its own left-hand column, and source and
# learner follow as two further columns, so every configuration is one row and
# the S- and T-learner sit in the same column. Repeated family and source names
# are left blank, which keeps the block structure visible without group-header
# rows.
#
# Other choices:
#   - Short family labels (DNN, XGBoost, BJ-ELM, BJ-trees); the full names are
#     in the caption, which is where a reader looks once.
#   - No variance column. Width and variance are near-deterministic transforms
#     of one another, and width is the one that can be weighed against the
#     effect scale.
#   - Only the lowest RMSE is bold. Bolding the narrowest interval would reward
#     exactly the overconfidence the table exposes, and "best coverage" is
#     ambiguous once a method overcovers.
#   - The fusion forest gets an em-dash in the learner column: it is not a
#     meta-learner, and a blank there would read as a missing value.
#   - No citations; the methods are cited in Section 3 where introduced.

fams <- c(deepAFT = "DNN", xgbAFT = "XGBoost",
          bjelm = "BJ-ELM", bjtree = "BJ-trees")

num <- function(x, d = 3) {
  if (length(x) == 0 || is.na(x)) return("---")
  s <- formatC(x, format = "f", digits = d)
  if (x < 0) paste0("$", s, "$") else s
}

s    <- tbl[tbl$population == "All", ]
get  <- function(m, col) { v <- s[[col]][s$method == m]; if (length(v)) v else NA_real_ }
best <- min(s$rmse, na.rm = TRUE)

cells <- function(m) {
  r  <- get(m, "rmse")
  rt <- num(r)
  if (!is.na(r) && isTRUE(all.equal(r, best))) rt <- paste0("$\\mathbf{", rt, "}$")
  paste(c(rt, num(get(m, "bias")), num(get(m, "coverage")),
          num(get(m, "width"), 2)), collapse = " & ")
}

rows <- c(sprintf("    Bayesian fusion forest & Both & --- & %s \\\\", cells("Fusion")),
          "    \\midrule")

for (f in names(fams)) {
  first_of_family <- TRUE
  for (src in c("RCT", "Pooled")) {
    lab <- if (src == "RCT") "Trial" else "Pool"
    for (l in c("S", "T")) {
      fam_cell <- if (first_of_family) fams[[f]] else ""
      src_cell <- if (l == "S") lab else ""
      rows <- c(rows, sprintf("    %-22s & %-5s & %s & %s \\\\",
                              fam_cell, src_cell, l, cells(paste0(f, "-", src, "-", l))))
      first_of_family <- FALSE
    }
  }
  if (f != names(fams)[length(fams)]) rows <- c(rows, "    \\addlinespace")
}

tex <- c(
"% Generated by simulations/exp2_competitor/combine_competitors.R -- do not edit.",
"\\begin{table}[!tb]",
"  \\centering",
"  \\small",
"  \\begin{tabular}{@{}lll rrrr@{}}",
"    \\toprule",
"    Method & Source & Learner & RMSE & Bias & Coverage & Width \\\\",
"    \\midrule",
rows,
"    \\bottomrule",
"  \\end{tabular}",
"  \\caption{Comparison with the machine-learning alternatives at",
"    $\\lambda_d = \\lambda_u = 1$, over the pooled evaluation sample. The",
"    alternatives are a deep neural network (DNN), gradient-boosted trees under",
"    an accelerated failure time loss (XGBoost), and Buckley--James boosting",
"    over extreme learning machines (BJ-ELM) and over regression trees",
"    (BJ-trees). Each is fitted as an S- and a T-learner, on the trial alone",
"    and on a naive pool of both sources; the fusion forest uses both sources",
"    and is not a meta-learner. Both sources are right-censored at circa",
"    $35\\%$, so every method sees the same data. Coverage and width refer to",
"    $95\\%$ intervals: posterior for the fusion forest, percentile bootstrap",
"    over $100$ resamples otherwise. Averages over $M = 1000$ Monte Carlo",
"    replications. The BJ-trees S-learner",
"    returns a coverage and width of exactly zero: componentwise boosting never",
"    selects the treatment indicator, because the treatment has no marginal",
"    effect in this data-generating process, so the estimated effect is",
"    identically zero in every replication and every bootstrap resample.}",
"  \\label{tab:sm-competitors}",
"\\end{table}")

texf <- file.path(res_dir, "competitors_table.tex")
writeLines(tex, texf)

cat("\nWritten:\n  ", csv, "\n  ", texf, "\n", sep = "")
