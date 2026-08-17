# ==============================================================================
# 3D case.R
# IISE Transactions second revision - theory-aligned 3D-printing benchmark
# ==============================================================================
#
#
# GPFR fitting note
#   Direct marginal-likelihood optimization and EM target the same GPFR
#   likelihood; this script uses direct MLE because WDEI additionally needs
#   cross-candidate latent covariance blocks that gpfrPredict() does not return.
#
# Statistical interpretation
#   Later curves are responses from one fixed data-calibrated digital environment,
#   not additional physical truth. Global RMSE, POG, and Optimal RMSE are
#   reference-relative benchmark measures.
#
# Data layouts
#   Preferred: three numeric CSV files in CASE3D_DATA_DIR:
#     * 15 x 3 scalar process inputs
#     * 15 x L fixed functional covariate/temperature trajectories
#     * 15 x L force-time response curves
#
# ==============================================================================

options(stringsAsFactors = FALSE, warn = 1, scipen = 6, digits = 6)

RUN_MODE <- tolower(Sys.getenv("CASE3D_RUN_MODE", unset = "paper"))
WINDOWS_DATA_DIR <- "C:/Users/dcf/Desktop/IISE_Code/3D Case/3D case data"
DEFAULT_DATA_DIR <- if (dir.exists(WINDOWS_DATA_DIR)) {
  WINDOWS_DATA_DIR
} else {
  file.path(getwd(), "3D case data")
}
DATA_DIR <- Sys.getenv("CASE3D_DATA_DIR", unset = DEFAULT_DATA_DIR)
OUTPUT_DIR <- Sys.getenv("CASE3D_OUTPUT_DIR", unset = file.path(getwd(), "3D case 16 outputs"))

CFG <- list(
  seed = 20260811L,
  data_dir = DATA_DIR,
  output_dir = OUTPUT_DIR,
  scalar_file = "",
  functional_covariate_file = "",
  response_file = "",

  input_names = c("Layer thickness", "Infilling rate", "Printing speed"),
  input_short = c("Layer", "Infill", "Speed"),
  input_lower = c(0.1, 0.1, 60),
  input_upper = c(0.3, 0.3, 80),

  M0 = 15L,
  N_total = 30L,
  candidate_size = 2000L,
  functional_points = 40L,
  time_min = 0.04,
  time_max = 20,

  n_basis = 9L,
  spline_degree = 3L,
  # A reduced common response-score basis avoids fitting separate covariance
  # amplitudes to near-null directions from only 15 curves.
  response_pc_variance = 0.995,
  response_pc_min = 3L,
  response_pc_max = 6L,
  ridge = 1e-7,
  covariance_floor = 1e-6,
  jitter = 1e-9,
  epsilon = 1e-12,

  # Use a calibrated value here whenever repeatability/sensor data are available.
  # NA estimates a homoscedastic iid noise SD jointly by GPFR marginal likelihood.
  measurement_sd_override = NA_real_,
  measurement_sd_floor_ratio = 0.0025,
  measurement_sd_cap_ratio = 0.20,

  # Fixed candidate-specific functional covariates. The default has the triangular
  # marginal stated for the case and deliberately adds no unreported AR(1) model.
  temp_lower = 202,
  temp_mode = 205,
  temp_upper = 208,

  # Joint GPFR covariance. Scalar inputs and supplied functional trajectories
  # enter the covariance as a product of powered-exponential kernels.
  kernel_gamma = 1.0,
  functional_kernel_gamma = 2.0,
  # The scalar printing parameters alone define the time-varying mean.
  analysis_trend = "linear",
  ell_u_bounds = c(0.03, 5.00),
  ell_x_bounds = c(0.05, 8.00),
  # Effective lower bound = max(ell_u_bounds[1], fraction * initial LHS
  # median nearest-neighbour distance). It is fixed before method comparison.
  ell_resolution_fraction = 0.50,
  signal_bounds = c(1e-4, 100),
  mle_initial_starts = 4L,
  mle_warm_starts = 2L,
  mle_maxit = 160L,

  # Functional-input FPCA is used only to define a stable finite-dimensional
  # L2 covariance distance between supplied trajectories. It is not a mean term.
  functional_pc_variance = 0.95,
  functional_pc_max = 3L,

  # A calibrated benchmark uses the same GPFR covariance family as the analysis
  # model. Independence comes from freezing the reference draw before any method,
  # not from deliberately misspecifying its kernel family.
  reference_kernel_gamma = 1.0,
  reference_rff = 512L,
  reference_draw_scale = 1.0,

  alpha = 0.50,
  kappa = 1.00,
  rho = 0.50,
  decay_at_r0 = 0.05,
  eqi_quantile = 0.90,

  one_shot_restarts = 1500L,
  diversity_batch = 64L,
  near_replication_fraction = 0.50,
  precompute_distances = TRUE,
  save_outputs = TRUE,
  show_plots = TRUE,
  verbose = TRUE
)

if (RUN_MODE == "smoke") {
  CFG$candidate_size <- 120L
  CFG$N_total <- 17L
  CFG$reference_rff <- 64L
  CFG$mle_initial_starts <- 2L
  CFG$mle_warm_starts <- 1L
  CFG$mle_maxit <- 50L
  CFG$one_shot_restarts <- 50L
  CFG$diversity_batch <- 30L
  CFG$save_outputs <- FALSE
  CFG$show_plots <- FALSE
}

if (RUN_MODE == "diagnostic") {
  CFG$candidate_size <- 500L
  CFG$N_total <- 30L
  CFG$reference_rff <- 160L
  CFG$mle_initial_starts <- 3L
  CFG$mle_warm_starts <- 1L
  CFG$mle_maxit <- 90L
  CFG$one_shot_restarts <- 300L
  CFG$diversity_batch <- 50L
  CFG$save_outputs <- FALSE
  # Diagnostic runs should still draw to the RStudio Plots pane.
  CFG$show_plots <- TRUE
}

METHODS <- c("Initial LHS", "One-shot LHS", "GPFR-EQI", "WDEI")

# ==============================================================================
# 1. Numerical utilities and randomized designs
# ==============================================================================

assert_true <- function(x, message) {
  if (!isTRUE(x)) stop(message, call. = FALSE)
}

with_seed <- function(seed, expr) {
  existed <- exists(".Random.seed", envir = .GlobalEnv, inherits = FALSE)
  if (existed) old <- get(".Random.seed", envir = .GlobalEnv)
  on.exit({
    if (existed) {
      assign(".Random.seed", old, envir = .GlobalEnv)
    } else if (exists(".Random.seed", envir = .GlobalEnv, inherits = FALSE)) {
      rm(".Random.seed", envir = .GlobalEnv)
    }
  }, add = TRUE)
  set.seed(as.integer(seed %% .Machine$integer.max))
  force(expr)
}

safe_chol <- function(A, jitter = 1e-10) {
  A <- (A + t(A)) / 2
  for (k in 0:11) {
    ans <- tryCatch(chol(A + diag(jitter * 10^k, nrow(A))),
                    error = function(e) NULL)
    if (!is.null(ans)) return(ans)
  }
  stop("Cholesky factorization failed after jitter stabilization.", call. = FALSE)
}

chol_solve <- function(U, B = NULL) {
  if (is.null(B)) return(chol2inv(U))
  backsolve(U, forwardsolve(t(U), B))
}

safe_solve <- function(A, B = NULL, jitter = 1e-10) {
  chol_solve(safe_chol(A, jitter), B)
}

logdet_chol <- function(U) 2 * sum(log(diag(U)))

sqdist <- function(A, B = A) {
  A <- as.matrix(A); B <- as.matrix(B)
  pmax(outer(rowSums(A^2), rowSums(B^2), "+") - 2 * tcrossprod(A, B), 0)
}

scale_to_unit <- function(U, lower, upper) {
  sweep(sweep(U, 2, lower, "-"), 2, upper - lower, "/")
}

lhs_unit <- function(n, d, seed = NULL) {
  draw <- function() {
    X <- matrix(NA_real_, n, d)
    for (j in seq_len(d)) X[, j] <- (sample.int(n) - runif(n)) / n
    X
  }
  if (is.null(seed)) draw() else with_seed(seed, draw())
}

min_pair_distance <- function(X) {
  D <- sqdist(X)
  diag(D) <- Inf
  sqrt(min(D))
}

conditional_lhs_targets <- function(initial_unit, n_add, restarts, seed) {
  with_seed(seed, {
    best <- NULL; best_score <- -Inf
    for (r in seq_len(restarts)) {
      candidate <- lhs_unit(n_add, ncol(initial_unit))
      score <- min_pair_distance(rbind(initial_unit, candidate))
      if (score > best_score) {
        best <- candidate; best_score <- score
      }
    }
    attr(best, "conditional_min_distance") <- best_score
    best
  })
}

map_targets_to_unique_pool <- function(targets, pool_unit, excluded, seed) {
  with_seed(seed, {
    available <- setdiff(seq_len(nrow(pool_unit)), unique(excluded))
    order_targets <- sample.int(nrow(targets))
    mapped <- integer(nrow(targets))
    for (k in order_targets) {
      d2 <- rowSums((pool_unit[available, , drop = FALSE] -
        matrix(targets[k, ], length(available), ncol(pool_unit), TRUE))^2)
      nearest <- available[which.min(d2)]
      mapped[k] <- nearest
      available <- setdiff(available, nearest)
    }
    mapped
  })
}

trapezoid_weights <- function(t) {
  n <- length(t)
  assert_true(n >= 2L && all(diff(t) > 0), "Time grid must be increasing.")
  w <- numeric(n)
  w[1] <- (t[2] - t[1]) / 2
  w[n] <- (t[n] - t[n - 1L]) / 2
  if (n > 2L) w[2:(n - 1L)] <- (t[3:n] - t[1:(n - 2L)]) / 2
  w
}

gaussian_ei <- function(mu, variance, incumbent) {
  sdv <- sqrt(pmax(variance, 0))
  delta <- incumbent - mu
  out <- pmax(delta, 0)
  ok <- sdv > 1e-12
  if (any(ok)) {
    z <- delta[ok] / sdv[ok]
    out[ok] <- delta[ok] * pnorm(z) + sdv[ok] * dnorm(z)
  }
  pmax(out, 0)
}

derive_eta <- function(U_unit, initial_idx, decay_at_r0) {
  D <- sqrt(sqdist(U_unit[initial_idx, , drop = FALSE]))
  diag(D) <- Inf
  r0 <- median(apply(D, 1, min))
  assert_true(is.finite(r0) && r0 > 0, "Cannot calibrate eta from initial design.")
  list(r0 = r0, eta = -log(decay_at_r0) / r0^2)
}

median_nearest_distance <- function(X, initial_idx) {
  if (!ncol(X)) return(1)
  D <- sqrt(sqdist(X[initial_idx, , drop = FALSE]))
  diag(D) <- Inf
  out <- median(apply(D, 1, min))
  assert_true(is.finite(out) && out > 0,
              "Cannot identify the functional-kernel resolution.")
  out
}

# ==============================================================================
# 2. Data import and deterministic self-test data
# ==============================================================================

read_csv_numeric <- function(path) {
  dat <- tryCatch(read.csv(path, check.names = FALSE), error = function(e) NULL)
  if (is.null(dat) || ncol(dat) <= 1L) {
    alt <- tryCatch(read.csv2(path, check.names = FALSE), error = function(e) NULL)
    if (!is.null(alt) && (is.null(dat) || ncol(alt) > ncol(dat))) dat <- alt
  }
  assert_true(!is.null(dat), paste("Cannot read", path))
  raw_names <- names(dat)
  nums <- lapply(dat, function(x) suppressWarnings(as.numeric(as.character(x))))
  keep <- vapply(nums, function(x) mean(is.finite(x)) >= 0.80, logical(1))
  assert_true(any(keep), paste("No numeric columns detected in", path))
  M <- as.matrix(as.data.frame(nums[keep], check.names = FALSE))
  colnames(M) <- raw_names[keep]
  M <- M[apply(M, 1, function(x) any(is.finite(x))), , drop = FALSE]
  if (ncol(M) > 1L) {
    first <- M[, 1]; nm <- tolower(colnames(M)[1])
    sequential <- all(is.finite(first)) &&
      (all(abs(first - seq_len(nrow(M))) < 1e-8) ||
       all(abs(first - (seq_len(nrow(M)) - 1L)) < 1e-8))
    if (sequential || grepl("^(x|id|index|sample|run|no\\.?)$", nm)) {
      M <- M[, -1, drop = FALSE]
    }
  }
  storage.mode(M) <- "double"
  M
}

read_response_txt <- function(path) {
  d <- tryCatch(read.table(path, header = TRUE, check.names = FALSE),
                error = function(e) NULL)
  if (is.null(d) || !nrow(d)) {
    d <- tryCatch(read.table(path, header = FALSE, check.names = FALSE),
                  error = function(e) NULL)
  }
  assert_true(!is.null(d) && nrow(d) >= 10L,
              paste("Cannot read response file:", path))
  numeric_cols <- which(vapply(d, is.numeric, logical(1)))
  assert_true(length(numeric_cols) > 0L, paste("No numeric response in", path))
  nms <- tolower(names(d)[numeric_cols])
  preferred <- grep("force|load|response|output", nms)
  chosen <- if (length(preferred)) numeric_cols[preferred[1L]] else {
    candidates <- numeric_cols
    if (length(candidates) > 1L) {
      first <- d[[candidates[1L]]]
      if (all(is.finite(first)) &&
          max(abs(first - seq_along(first))) < 1e-8) candidates <- candidates[-1L]
    }
    candidates[1L]
  }
  y <- as.numeric(d[[chosen]])
  assert_true(all(is.finite(y)), paste("Non-finite response in", path))
  y
}

orient_functional <- function(M, n_runs) {
  if (nrow(M) == n_runs) return(M)
  if (ncol(M) == n_runs) return(t(M))
  stop("Functional CSV cannot be oriented to the initial runs.", call. = FALSE)
}

extract_scalar <- function(M, n_runs) {
  if (ncol(M) == n_runs && nrow(M) <= 6L) M <- t(M)
  assert_true(nrow(M) == n_runs, "Scalar-input CSV has the wrong number of runs.")
  nms <- tolower(colnames(M))
  keys <- c("layer|thickness|height", "infill", "speed")
  idx <- vapply(keys, function(k) {
    hit <- grep(k, nms); if (length(hit)) hit[1] else NA_integer_
  }, integer(1))
  if (all(is.finite(idx))) return(M[, idx, drop = FALSE])
  assert_true(ncol(M) >= 3L, "Scalar-input CSV needs three numeric columns.")
  M[, seq_len(3L), drop = FALSE]
}

triangular_quantile <- function(p, lower, mode, upper) {
  p <- pmin(pmax(p, 1e-12), 1 - 1e-12)
  cut <- (mode - lower) / (upper - lower)
  ifelse(p < cut,
         lower + sqrt(p * (upper - lower) * (mode - lower)),
         upper - sqrt((1 - p) * (upper - lower) * (upper - mode)))
}

generate_fixed_functional_covariates <- function(n, n_time, cfg, seed) {
  with_seed(seed, {
    P <- matrix(runif(n * n_time), n, n_time)
    round(triangular_quantile(P, cfg$temp_lower, cfg$temp_mode,
                              cfg$temp_upper), 1)
  })
}

discover_data_files <- function(cfg) {
  explicit <- c(cfg$scalar_file, cfg$functional_covariate_file,
                cfg$response_file)
  if (all(nzchar(explicit))) {
    paths <- ifelse(file.exists(explicit), explicit,
                    file.path(cfg$data_dir, explicit))
    assert_true(all(file.exists(paths)), "One or more configured data files are missing.")
    return(setNames(as.list(paths), c("scalar", "functional", "response")))
  }
  assert_true(dir.exists(cfg$data_dir), paste0(
    "Data directory does not exist: ", cfg$data_dir,
    "\nSet CASE3D_DATA_DIR or edit CFG$data_dir."
  ))
  csv <- list.files(cfg$data_dir, "\\.csv$", full.names = TRUE,
                    recursive = TRUE, ignore.case = TRUE)
  assert_true(length(csv) >= 3L, "At least three numeric CSV files are required.")
  mats <- lapply(csv, read_csv_numeric)
  base <- tolower(basename(csv))
  scalar_score <- 10 * as.numeric(grepl(
    "scalar|parameter|process|train.*frgp|input.*u", base
  )) + 5 * vapply(mats, function(M) as.numeric(
    (nrow(M) == cfg$M0 && ncol(M) %in% 3:6) ||
      (ncol(M) == cfg$M0 && nrow(M) %in% 3:6)), numeric(1))
  scalar_i <- which.max(scalar_score)
  assert_true(scalar_score[scalar_i] > 0, "Cannot identify scalar-input CSV.")
  remaining <- setdiff(seq_along(csv), scalar_i)
  fun <- lapply(mats[remaining], function(M) tryCatch(
    orient_functional(M, cfg$M0), error = function(e) NULL
  ))
  valid <- which(vapply(fun, function(M) !is.null(M) && ncol(M) >= 10L,
                        logical(1)))
  assert_true(length(valid) >= 2L, "Cannot identify functional input and response CSVs.")
  remaining <- remaining[valid]; fun <- fun[valid]
  temp_score <- 10 * as.numeric(grepl(
    "temp|temperature|nozzle|functional.*input", base[remaining]
  )) + 5 * vapply(fun, function(M) {
    med <- median(M, na.rm = TRUE)
    as.numeric(is.finite(med) && med >= 195 && med <= 215)
  }, numeric(1))
  temp_local <- which.max(temp_score)
  assert_true(temp_score[temp_local] > 0, "Cannot identify functional-input CSV.")
  temp_i <- remaining[temp_local]
  response_pool <- setdiff(remaining, temp_i)
  response_score <- as.numeric(grepl(
    "force|response|output|curve|functional.*response", base[response_pool]
  ))
  response_i <- response_pool[which.max(response_score + 1)]
  list(scalar = csv[scalar_i], functional = csv[temp_i],
       response = csv[response_i])
}

load_legacy_layout <- function(cfg) {
  if (!dir.exists(cfg$data_dir)) return(NULL)
  csv <- list.files(cfg$data_dir, "\\.csv$", full.names = TRUE,
                    recursive = TRUE, ignore.case = TRUE)
  txt <- list.files(cfg$data_dir, "^[0-9]+\\.txt$", full.names = TRUE,
                    recursive = TRUE, ignore.case = TRUE)
  if (!length(csv) || length(txt) < cfg$M0) return(NULL)
  number <- suppressWarnings(as.integer(tools::file_path_sans_ext(basename(txt))))
  ord <- order(number); txt <- txt[ord]; number <- number[ord]
  txt <- txt[is.finite(number)][seq_len(cfg$M0)]
  mats <- lapply(csv, read_csv_numeric)
  base <- tolower(basename(csv))
  scalar_score <- 10 * as.numeric(grepl("train.*frgp|scalar|parameter|input", base)) +
    5 * vapply(mats, function(M) as.numeric(
      (nrow(M) == cfg$M0 && ncol(M) >= 3L) ||
        (ncol(M) == cfg$M0 && nrow(M) >= 3L)), numeric(1))
  scalar_i <- which.max(scalar_score)
  if (!length(scalar_i) || scalar_score[scalar_i] <= 0) return(NULL)
  U <- extract_scalar(mats[[scalar_i]], cfg$M0)
  curves <- lapply(txt, read_response_txt)
  common_length <- min(vapply(curves, length, integer(1)))
  assert_true(common_length >= 10L, "Legacy response curves are too short.")
  keep <- unique(round(seq(1, common_length,
                           length.out = min(cfg$functional_points,
                                            common_length))))
  Y <- t(vapply(curves, function(y) y[keep], numeric(length(keep))))
  # The original 3D-printing protocol defines one fixed triangular nozzle-
  # temperature trajectory for each run. Reproduce those planned covariates
  # deterministically when the compact legacy files do not store them. They are
  # generated once here and are never regenerated during method comparison.
  X <- generate_fixed_functional_covariates(
    cfg$M0, length(keep), cfg, cfg$seed + 13L
  )
  colnames(U) <- cfg$input_short
  list(U = U, X = X, Y = Y,
       files = list(scalar = csv[scalar_i], functional = "unavailable",
                    response = txt),
       functional_observed = TRUE,
       functional_source_initial =
         "Fixed Triangular(202,205,208) trajectory from case protocol")
}

make_self_test_data <- function(cfg) {
  with_seed(cfg$seed + 11L, {
    Uu <- lhs_unit(cfg$M0, 3L)
    U <- sweep(sweep(Uu, 2, cfg$input_upper - cfg$input_lower, "*"),
               2, cfg$input_lower, "+")
    L <- 40L; t <- seq(cfg$time_min, cfg$time_max, length.out = L)
    X <- generate_fixed_functional_covariates(cfg$M0, L, cfg, cfg$seed + 13L)
    Y <- matrix(NA_real_, cfg$M0, L)
    for (i in seq_len(cfg$M0)) {
      trend <- 1.8 + 0.9 * Uu[i, 2] - 0.35 * Uu[i, 1] +
        0.30 * sin(pi * t / max(t)) +
        0.18 * Uu[i, 3] * cos(2 * pi * t / max(t)) +
        0.08 * Uu[i, 1] * Uu[i, 2]
      Y[i, ] <- trend + 0.045 * (X[i, ] - cfg$temp_mode) +
        rnorm(L, 0, 0.025)
    }
    colnames(U) <- cfg$input_short
    list(U = U, X = X, Y = Y,
         files = list(scalar = "self-test", functional = "self-test",
                      response = "self-test"),
         functional_observed = TRUE,
         functional_source_initial = "Generated self-test trajectory")
  })
}

load_case_data <- function(cfg) {
  if (RUN_MODE %in% c("smoke", "diagnostic") && !dir.exists(cfg$data_dir)) {
    return(make_self_test_data(cfg))
  }
  legacy <- load_legacy_layout(cfg)
  csv_n <- if (dir.exists(cfg$data_dir)) length(list.files(
    cfg$data_dir, "\\.csv$", recursive = TRUE, ignore.case = TRUE)) else 0L
  if (csv_n < 3L && !is.null(legacy)) return(legacy)
  files <- discover_data_files(cfg)
  U <- extract_scalar(read_csv_numeric(files$scalar), cfg$M0)
  X <- orient_functional(read_csv_numeric(files$functional), cfg$M0)
  Y <- orient_functional(read_csv_numeric(files$response), cfg$M0)
  assert_true(ncol(X) == ncol(Y), "Functional-input and response grids differ.")
  if (ncol(Y) > cfg$functional_points) {
    keep <- unique(round(seq(1, ncol(Y), length.out = cfg$functional_points)))
    X <- X[, keep, drop = FALSE]; Y <- Y[, keep, drop = FALSE]
  }
  assert_true(all(is.finite(U)) && all(is.finite(X)) && all(is.finite(Y)),
              "Input files contain missing or non-finite values.")
  inside <- sweep(U, 2, cfg$input_lower, ">=") &
    sweep(U, 2, cfg$input_upper, "<=")
  assert_true(all(inside), "Observed scalar inputs fall outside configured bounds.")
  colnames(U) <- cfg$input_short
  list(U = U, X = X, Y = Y, files = files,
       functional_observed = TRUE,
       functional_source_initial = "Supplied physical functional trajectory")
}

# ==============================================================================
# 3. Candidate library, spline basis, and fixed objective
# ==============================================================================

feature_map <- function(U_unit, type = "quadratic") {
  U_unit <- as.matrix(U_unit)
  linear <- cbind(Intercept = 1, U1 = U_unit[, 1],
                  U2 = U_unit[, 2], U3 = U_unit[, 3])
  if (type == "linear") return(linear)
  squares <- U_unit^2
  colnames(squares) <- c("U1_sq", "U2_sq", "U3_sq")
  if (type == "quadratic") return(cbind(linear, squares))
  if (type == "quadratic_interaction") {
    interactions <- cbind(
      U1_U2 = U_unit[, 1] * U_unit[, 2],
      U1_U3 = U_unit[, 1] * U_unit[, 3],
      U2_U3 = U_unit[, 2] * U_unit[, 3]
    )
    return(cbind(linear, squares, interactions))
  }
  stop("Unknown trend type: ", type, call. = FALSE)
}

gpfr_feature_map <- function(case, idx) {
  # Revised GPFR: scalar inputs define the time-varying mean. Functional-input
  # coordinates are deliberately excluded here and enter candidate_kernel().
  feature_map(case$U_unit[idx, , drop = FALSE],
              case$cfg$analysis_trend)
}

make_basis <- function(t, cfg) {
  B <- splines::bs(t, df = cfg$n_basis, degree = cfg$spline_degree,
                   intercept = TRUE, Boundary.knots = range(t))
  P <- safe_solve(crossprod(B) + diag(cfg$ridge, ncol(B)),
                  t(B), cfg$jitter)
  q <- trapezoid_weights(t); q <- q / sum(q)
  coefficient_objective <- as.vector(crossprod(B, -q))
  list(B = B, P = P, H = ncol(B), q = q,
       coefficient_objective = coefficient_objective)
}

estimate_measurement_sd_start <- function(Y, basis, cfg) {
  C <- Y %*% t(basis$P)
  residual <- Y - C %*% t(basis$B)
  ysd <- max(sd(as.vector(Y)), 1e-8)
  raw <- sqrt(mean(residual^2))
  lower <- cfg$measurement_sd_floor_ratio * ysd
  upper <- cfg$measurement_sd_cap_ratio * ysd
  value <- if (is.finite(cfg$measurement_sd_override)) {
    cfg$measurement_sd_override
  } else min(max(raw, lower), upper)
  list(value = value, lower = lower, upper = upper, raw_spline_sd = raw,
       source = if (is.finite(cfg$measurement_sd_override))
         "user/calibration override" else "MLE start from spline residual")
}

build_case <- function(data, cfg) {
  L <- ncol(data$Y); t <- seq(cfg$time_min, cfg$time_max, length.out = L)
  basis <- make_basis(t, cfg)
  n_new <- cfg$candidate_size - cfg$M0
  assert_true(n_new >= 0L, "candidate_size must be at least M0.")
  U_new_unit <- lhs_unit(n_new, 3L, cfg$seed + 101L)
  U_new <- sweep(sweep(U_new_unit, 2, cfg$input_upper - cfg$input_lower, "*"),
                 2, cfg$input_lower, "+")
  U <- rbind(data$U, U_new)
  U_unit <- scale_to_unit(U, cfg$input_lower, cfg$input_upper)
  if (isTRUE(data$functional_observed)) {
    X_new <- generate_fixed_functional_covariates(
      n_new, L, cfg, cfg$seed + 211L
    )
    X <- rbind(data$X, X_new)
    # Basis smoothing and FPCA provide stable coordinates for the functional
    # trajectory covariance kernel. No response information is used here.
    X_coef <- X %*% t(basis$P)
    x_center <- colMeans(X_coef[seq_len(cfg$M0), , drop = FALSE])
    X_centered <- sweep(X_coef, 2, x_center, "-")
    cov_x <- crossprod(X_centered[seq_len(cfg$M0), , drop = FALSE]) /
      max(cfg$M0 - 1L, 1L)
    eig_x <- eigen((cov_x + t(cov_x)) / 2, symmetric = TRUE)
    positive <- pmax(eig_x$values, 0)
    if (sum(positive) <= cfg$epsilon) {
      k_x <- 0L
    } else {
      k_x <- which(cumsum(positive) / sum(positive) >=
                     cfg$functional_pc_variance)[1L]
      k_x <- min(k_x, cfg$functional_pc_max, cfg$M0 - 2L)
    }
    if (k_x > 0L) {
      X_rotation <- eig_x$vectors[, seq_len(k_x), drop = FALSE]
      X_score_scale <- sqrt(pmax(positive[seq_len(k_x)], cfg$epsilon))
      X_scores <- sweep(X_centered %*% X_rotation, 2, X_score_scale, "/")
    } else {
      X_rotation <- matrix(numeric(), ncol(X_coef), 0L)
      X_score_scale <- numeric()
      X_scores <- matrix(numeric(), nrow(X), 0L)
    }
  } else {
    # Defensive fallback for user-defined data loaders. The built-in preferred
    # and legacy loaders both supply fixed functional trajectories.
    X <- matrix(cfg$temp_mode, nrow(U), L)
    X_coef <- X %*% t(basis$P)
    x_center <- colMeans(X_coef[seq_len(cfg$M0), , drop = FALSE])
    X_rotation <- matrix(numeric(), ncol(X_coef), 0L)
    X_score_scale <- numeric()
    X_scores <- matrix(numeric(), nrow(U), 0L)
  }
  eta <- derive_eta(U_unit, seq_len(cfg$M0), cfg$decay_at_r0)
  cfg$r0 <- eta$r0; cfg$eta <- eta$eta
  cfg$r0_x <- median_nearest_distance(X_scores, seq_len(cfg$M0))
  cfg$ell_u_requested_bounds <- cfg$ell_u_bounds
  cfg$ell_x_requested_bounds <- cfg$ell_x_bounds
  cfg$ell_u_bounds[1] <- max(
    cfg$ell_u_bounds[1], cfg$ell_resolution_fraction * cfg$r0
  )
  cfg$ell_x_bounds[1] <- max(
    cfg$ell_x_bounds[1], cfg$ell_resolution_fraction * cfg$r0_x
  )
  assert_true(cfg$ell_u_bounds[1] < cfg$ell_u_bounds[2],
              "Identifiable scalar length-scale bounds have zero width.")
  assert_true(cfg$ell_x_bounds[1] < cfg$ell_x_bounds[2],
              "Identifiable functional length-scale bounds have zero width.")
  noise_start <- estimate_measurement_sd_start(data$Y, basis, cfg)
  case <- list(cfg = cfg, data = data, t = t, basis = basis,
               U = U, U_unit = U_unit, X = X, X_coef = X_coef,
               X_scores = X_scores, X_rotation = X_rotation,
               initial_idx = seq_len(cfg$M0), initial_Y = data$Y,
               x_center = x_center, x_score_scale = X_score_scale,
               noise_start = noise_start)
  if (isTRUE(cfg$precompute_distances)) {
    case$Du_all <- sqdist(U_unit)
    case$Dx_all <- sqdist(X_scores)
  }
  case
}

# ==============================================================================
# 4. Basis-space GPFR fitted by marginal maximum likelihood
# ==============================================================================

initialize_coregionalization <- function(case) {
  Y <- case$initial_Y
  idx_obs <- case$initial_idx
  measurement_sd_start <- case$noise_start$value
  C <- Y %*% t(case$basis$P)
  G <- gpfr_feature_map(case, idx_obs)
  beta <- safe_solve(crossprod(G) + diag(case$cfg$ridge, ncol(G)),
                     crossprod(G, C), case$cfg$jitter)
  residual <- C - G %*% beta
  df <- max(nrow(residual) - qr(G)$rank, 1L)
  empirical <- crossprod(residual) / df

  Sigma_e_coef_unit <- case$basis$P %*% t(case$basis$P)
  Sigma_e_coef_unit <- (Sigma_e_coef_unit + t(Sigma_e_coef_unit)) / 2
  latent <- empirical - measurement_sd_start^2 * Sigma_e_coef_unit
  latent <- (latent + t(latent)) / 2
  eg0 <- eigen(latent, symmetric = TRUE)
  floor_value <- case$cfg$covariance_floor *
    max(max(eg0$values), mean(diag(empirical)), 1e-10)
  latent_psd <- eg0$vectors %*%
    diag(pmax(eg0$values, floor_value), length(eg0$values)) %*%
    t(eg0$vectors)

  # Generalized functional principal components make the latent scores
  # independent while keeping the projected iid measurement covariance diagonal.
  L_e <- t(safe_chol(Sigma_e_coef_unit +
                       diag(case$cfg$jitter, nrow(Sigma_e_coef_unit)),
                     case$cfg$jitter))
  W_e <- solve(L_e)
  generalized <- W_e %*% latent_psd %*% t(W_e)
  generalized <- (generalized + t(generalized)) / 2
  eg <- eigen(generalized, symmetric = TRUE)
  lam_floor <- case$cfg$covariance_floor * max(eg$values[1], 1)
  lam_all <- pmax(eg$values, lam_floor)
  explained <- cumsum(pmax(eg$values, 0)) /
    max(sum(pmax(eg$values, 0)), case$cfg$epsilon)
  k_response <- which(explained >= case$cfg$response_pc_variance)[1L]
  if (!is.finite(k_response)) k_response <- length(lam_all)
  k_response <- max(case$cfg$response_pc_min,
                    min(k_response, case$cfg$response_pc_max,
                        length(lam_all), case$cfg$M0 - 2L))
  lam <- lam_all[seq_len(k_response)]
  V <- eg$vectors[, seq_len(k_response), drop = FALSE]
  A <- L_e %*% V %*% diag(sqrt(lam), length(lam))
  # Left inverse of the rectangular reduced-rank loading matrix.
  Ainv <- diag(1 / sqrt(lam), length(lam)) %*% t(V) %*% W_e
  score_noise_full_unit <- Ainv %*% Sigma_e_coef_unit %*% t(Ainv)
  score_noise_unit <- pmax(diag(score_noise_full_unit), 1e-12)
  Z <- C %*% t(Ainv)
  curve_loading <- case$basis$B %*% A
  objective_loading <- as.vector(t(A) %*% case$basis$coefficient_objective)
  integrated_weights <- colSums(curve_loading^2 * case$basis$q)
  offdiag <- score_noise_full_unit - diag(diag(score_noise_full_unit))
  list(A = A, Ainv = Ainv, Z = Z,
       score_noise_unit = score_noise_unit,
       score_noise_offdiag_ratio = sum(abs(offdiag)) /
         max(sum(abs(score_noise_full_unit)), 1e-12),
       response_components = k_response,
       response_variance_explained = explained[k_response],
       curve_loading = curve_loading,
       objective_loading = objective_loading,
       integrated_weights = integrated_weights,
       latent_coefficient_covariance = A %*% t(A))
}

transform_to_common_scores <- function(Y, case) {
  coreg <- case$coreg_template
  C <- Y %*% t(case$basis$P)
  coreg$Z <- C %*% t(coreg$Ainv)
  coreg
}

scalar_kernel_distance2 <- function(case, idx_a, idx_b, ell_u) {
  if (!is.null(case$Du_all)) {
    return(pmax(case$Du_all[idx_a, idx_b, drop = FALSE] / ell_u^2, 0))
  }
  pmax(sqdist(case$U_unit[idx_a, , drop = FALSE],
              case$U_unit[idx_b, , drop = FALSE]) / ell_u^2, 0)
}

functional_kernel_distance2 <- function(case, idx_a, idx_b, ell_x) {
  if (!ncol(case$X_scores)) {
    return(matrix(0, length(idx_a), length(idx_b)))
  }
  if (!is.null(case$Dx_all)) {
    return(pmax(case$Dx_all[idx_a, idx_b, drop = FALSE] / ell_x^2, 0))
  }
  pmax(sqdist(case$X_scores[idx_a, , drop = FALSE],
              case$X_scores[idx_b, , drop = FALSE]) / ell_x^2, 0)
}

candidate_kernel <- function(case, idx_a, idx_b, ell_u, ell_x) {
  # Product covariance on scalar inputs and supplied functional trajectories.
  # The trajectory term is a finite-dimensional L2 kernel obtained after the
  # fixed basis/FPCA approximation permitted in the revised GPFR formulation.
  du2 <- scalar_kernel_distance2(case, idx_a, idx_b, ell_u)
  dx2 <- functional_kernel_distance2(case, idx_a, idx_b, ell_x)
  exp(-pmax(du2, 0)^(case$cfg$kernel_gamma / 2) -
        pmax(dx2, 0)^(case$cfg$functional_kernel_gamma / 2))
}

gpfr_parameter_bounds <- function(H, case) {
  noise_lower <- if (is.finite(case$cfg$measurement_sd_override)) {
    case$cfg$measurement_sd_override
  } else case$noise_start$lower
  noise_upper <- if (is.finite(case$cfg$measurement_sd_override)) {
    case$cfg$measurement_sd_override
  } else case$noise_start$upper
  assert_true(is.finite(noise_lower) && is.finite(noise_upper) &&
                noise_lower > 0 && noise_upper > 0,
              "Measurement-noise bounds must be positive.")
  # L-BFGS-B needs nonzero width. A calibrated override is held essentially fixed.
  if (noise_upper <= noise_lower) {
    noise_lower <- noise_lower * (1 - 1e-9)
    noise_upper <- noise_upper * (1 + 1e-9)
  }
  lower <- log(c(case$cfg$ell_u_bounds[1], case$cfg$ell_x_bounds[1],
                 noise_lower,
                 rep(case$cfg$signal_bounds[1], H)))
  upper <- log(c(case$cfg$ell_u_bounds[2], case$cfg$ell_x_bounds[2],
                 noise_upper,
                 rep(case$cfg$signal_bounds[2], H)))
  list(lower = lower, upper = upper)
}

decode_gpfr_parameters <- function(par, H) {
  value <- exp(par)
  list(ell_u = value[1], ell_x = value[2], sigma_e = value[3],
       signal = value[3 + seq_len(H)])
}

gpfr_nll <- function(par, Z, idx_obs, case, coreg) {
  Hn <- ncol(Z); hp <- decode_gpfr_parameters(par, Hn)
  F <- gpfr_feature_map(case, idx_obs)
  R <- candidate_kernel(case, idx_obs, idx_obs, hp$ell_u, hp$ell_x)
  nll <- 0
  for (h in seq_len(Hn)) {
    noise_h <- hp$sigma_e^2 * coreg$score_noise_unit[h]
    K <- hp$signal[h] * R + diag(noise_h, nrow(R))
    Uc <- tryCatch(safe_chol(K, case$cfg$jitter), error = function(e) NULL)
    if (is.null(Uc)) return(.Machine$double.xmax / 100)
    KiF <- chol_solve(Uc, F)
    Cbeta <- tryCatch(safe_solve(crossprod(F, KiF) +
                                  diag(case$cfg$ridge, ncol(F)),
                                jitter = case$cfg$jitter),
                      error = function(e) NULL)
    if (is.null(Cbeta)) return(.Machine$double.xmax / 100)
    beta <- Cbeta %*% crossprod(F, chol_solve(Uc, Z[, h]))
    residual <- Z[, h] - as.vector(F %*% beta)
    nll <- nll + 0.5 * (logdet_chol(Uc) +
      sum(residual * chol_solve(Uc, residual)) +
      length(residual) * log(2 * pi))
  }
  if (!is.finite(nll)) .Machine$double.xmax / 100 else nll
}

initial_gpfr_parameter <- function(H, case, warm_model = NULL) {
  if (!is.null(warm_model) && length(warm_model$signal) == H) {
    return(log(c(warm_model$ell_u, warm_model$ell_x,
                 warm_model$sigma_e,
                 warm_model$signal)))
  }
  # Start at the experimentally resolvable scale, not at a hard-coded value.
  ell_start <- min(
    max(case$cfg$r0, 1.25 * case$cfg$ell_u_bounds[1]),
    0.80 * case$cfg$ell_u_bounds[2]
  )
  ell_x_start <- min(
    max(case$cfg$r0_x, 1.25 * case$cfg$ell_x_bounds[1]),
    0.80 * case$cfg$ell_x_bounds[2]
  )
  log(c(ell_start, ell_x_start, case$noise_start$value, rep(1, H)))
}

fit_gpfr_mle <- function(Y, idx_obs, case, warm_model = NULL,
                         role = "analysis") {
  coreg <- transform_to_common_scores(Y, case)
  Z <- coreg$Z; Hn <- ncol(Z)
  bounds <- gpfr_parameter_bounds(Hn, case)
  base <- pmin(pmax(initial_gpfr_parameter(Hn, case, warm_model),
                    bounds$lower), bounds$upper)
  n_starts <- if (is.null(warm_model)) case$cfg$mle_initial_starts else
    case$cfg$mle_warm_starts
  starts <- vector("list", n_starts)
  starts[[1]] <- base
  if (n_starts > 1L) {
    starts[2:n_starts] <- with_seed(
      case$cfg$seed + length(idx_obs) * 1009L +
        ifelse(role == "reference", 7001L, 0L),
      lapply(seq_len(n_starts - 1L), function(i) {
        pmin(pmax(base + rnorm(length(base), 0, 0.30),
                  bounds$lower), bounds$upper)
      })
    )
  }
  fits <- lapply(starts, function(start) tryCatch(
    optim(start, gpfr_nll, Z = Z, idx_obs = idx_obs, case = case,
          coreg = coreg, method = "L-BFGS-B", lower = bounds$lower,
          upper = bounds$upper,
          control = list(maxit = case$cfg$mle_maxit, factr = 1e7)),
    error = function(e) NULL
  ))
  valid <- which(vapply(fits, function(x) !is.null(x) && is.finite(x$value),
                         logical(1)))
  assert_true(length(valid) > 0L, "All GPFR marginal-likelihood fits failed.")
  best <- fits[[valid[which.min(vapply(fits[valid], `[[`, numeric(1), "value"))]]]
  hp <- decode_gpfr_parameters(best$par, Hn)
  F <- gpfr_feature_map(case, idx_obs)
  R <- candidate_kernel(case, idx_obs, idx_obs, hp$ell_u, hp$ell_x)
  score_fits <- vector("list", Hn)
  for (h in seq_len(Hn)) {
    noise_h <- hp$sigma_e^2 * coreg$score_noise_unit[h]
    K <- hp$signal[h] * R + diag(noise_h, nrow(R))
    Uc <- safe_chol(K, case$cfg$jitter)
    Kinv <- chol_solve(Uc)
    Cbeta <- safe_solve(crossprod(F, Kinv %*% F) +
                          diag(case$cfg$ridge, ncol(F)),
                        jitter = case$cfg$jitter)
    beta <- Cbeta %*% crossprod(F, Kinv %*% Z[, h])
    residual <- Z[, h] - as.vector(F %*% beta)
    score_fits[[h]] <- list(signal = hp$signal[h], noise = noise_h,
                            beta = beta, alpha = Kinv %*% residual,
                            Kinv = Kinv, F_train = F, Cbeta = Cbeta)
  }
  at_lower <- abs(best$par - bounds$lower) < 1e-5
  at_upper <- abs(best$par - bounds$upper) < 1e-5
  list(role = role, idx_obs = idx_obs, Y_obs = Y, Z_obs = Z,
       coreg = coreg, ell_u = hp$ell_u, ell_x = hp$ell_x,
       sigma_e = hp$sigma_e, signal = hp$signal,
       score_fits = score_fits, nll = best$value,
       convergence = best$convergence, message = best$message,
       ell_at_boundary = at_lower[1] || at_upper[1],
       ell_x_at_boundary = at_lower[2] || at_upper[2],
       noise_at_boundary = at_lower[3] || at_upper[3],
       signal_at_boundary = any(at_lower[-c(1, 2, 3)] |
                                  at_upper[-c(1, 2, 3)]),
       parameter_at_boundary = any(at_lower | at_upper),
       case = case)
}

predict_gpfr_scores <- function(model, idx) {
  case <- model$case
  Fs <- gpfr_feature_map(case, idx)
  Rst <- candidate_kernel(case, idx, model$idx_obs,
                          model$ell_u, model$ell_x)
  mu <- variance <- matrix(NA_real_, length(idx), length(model$score_fits))
  for (h in seq_along(model$score_fits)) {
    fit <- model$score_fits[[h]]
    Kst <- fit$signal * Rst
    mu[, h] <- as.vector(Fs %*% fit$beta + Kst %*% fit$alpha)
    base <- rowSums((Kst %*% fit$Kinv) * Kst)
    trend_residual <- Fs - Kst %*% fit$Kinv %*% fit$F_train
    trend_var <- rowSums((trend_residual %*% fit$Cbeta) * trend_residual)
    variance[, h] <- pmax(fit$signal - base + trend_var, 1e-14)
  }
  list(mu = mu, variance = variance)
}

predict_gpfr <- function(model, idx = seq_len(nrow(model$case$U)),
                         return_curve = TRUE) {
  scores <- predict_gpfr_scores(model, idx)
  g <- model$coreg$objective_loading
  out <- list(
    idx = idx, mu_score = scores$mu, var_score = scores$variance,
    muJ = as.vector(scores$mu %*% g),
    varJ = pmax(as.vector(scores$variance %*% (g^2)), 1e-14),
    integrated = as.vector(scores$variance %*%
                             model$coreg$integrated_weights)
  )
  if (return_curve) {
    out$mu_curve <- scores$mu %*% t(model$coreg$curve_loading)
    out$var_curve <- scores$variance %*% t(model$coreg$curve_loading^2)
  }
  out
}

posterior_crosscov_block <- function(model, h, idx_all, idx_batch,
                                     precomputed = NULL) {
  case <- model$case
  fit <- model$score_fits[[h]]
  if (is.null(precomputed)) {
    R_all_train <- candidate_kernel(case, idx_all, model$idx_obs,
                                    model$ell_u, model$ell_x)
    K_all_train <- fit$signal * R_all_train
    M_all <- K_all_train %*% fit$Kinv
    F_all <- gpfr_feature_map(case, idx_all)
    trend_all <- F_all - M_all %*% fit$F_train
  } else {
    K_all_train <- precomputed$K_all_train
    M_all <- precomputed$M_all
    trend_all <- precomputed$trend_all
  }
  R_ab <- candidate_kernel(case, idx_all, idx_batch,
                           model$ell_u, model$ell_x)
  K_batch_train <- fit$signal * candidate_kernel(
    case, idx_batch, model$idx_obs, model$ell_u, model$ell_x
  )
  F_batch <- gpfr_feature_map(case, idx_batch)
  trend_batch <- F_batch - K_batch_train %*% fit$Kinv %*% fit$F_train
  fit$signal * R_ab - M_all %*% t(K_batch_train) +
    trend_all %*% fit$Cbeta %*% t(trend_batch)
}

# ==============================================================================
# 5. Fixed joint posterior reference world without 200 pseudo responses
# ==============================================================================

reference_feature_matrix <- function(case, model, n_features, seed) {
  # Independent spectral frequencies for the scalar and functional factors give
  # random features for their product covariance. The seed is fixed before any
  # comparison method runs, so this reference path is exactly reproducible.
  Zu <- case$U_unit / model$ell_u
  Zx <- case$X_scores / model$ell_x
  draw_frequency <- function(n_features, dimension, gamma) {
    if (!dimension) return(matrix(0, n_features, 0L))
    if (abs(gamma - 1) < 1e-10) {
      denom <- sqrt(rchisq(n_features, df = 1))
      matrix(rnorm(n_features * dimension), n_features, dimension) / denom
    } else if (abs(gamma - 2) < 1e-10) {
      matrix(rnorm(n_features * dimension), n_features, dimension) * sqrt(2)
    } else {
      stop("Reference RFF supports powered-exponential gamma 1 or 2.",
           call. = FALSE)
    }
  }
  with_seed(seed, {
    Wu <- draw_frequency(n_features, ncol(Zu), case$cfg$kernel_gamma)
    Wx <- draw_frequency(n_features, ncol(Zx),
                         case$cfg$functional_kernel_gamma)
    phase <- runif(n_features, 0, 2 * pi)
    linear <- Zu %*% t(Wu)
    if (ncol(Zx)) linear <- linear + Zx %*% t(Wx)
    sqrt(2 / n_features) * cos(linear +
      matrix(phase, nrow(Zu), n_features, byrow = TRUE))
  })
}

joint_reference_draw <- function(reference_model, case) {
  all_idx <- seq_len(nrow(case$U)); train <- reference_model$idx_obs
  Phi <- reference_feature_matrix(
    case, reference_model, case$cfg$reference_rff, case$cfg$seed + 4019L
  )
  Phi_train <- Phi[train, , drop = FALSE]
  F_all <- gpfr_feature_map(case, all_idx)
  draw <- matrix(NA_real_, nrow(case$U), length(reference_model$score_fits))
  with_seed(case$cfg$seed + 4049L, {
    for (h in seq_along(reference_model$score_fits)) {
      fit <- reference_model$score_fits[[h]]
      theta <- rnorm(ncol(Phi))
      prior_all <- sqrt(fit$signal) * as.vector(Phi %*% theta)
      simulated_noise <- rnorm(length(train), 0, sqrt(fit$noise))
      K_train <- fit$signal * tcrossprod(Phi_train) +
        diag(fit$noise, length(train))
      K_all_train <- fit$signal * Phi %*% t(Phi_train)
      observed_residual <- reference_model$Z_obs[, h] -
        as.vector(fit$F_train %*% fit$beta)
      observed_alpha <- safe_solve(K_train, observed_residual,
                                   case$cfg$jitter)
      simulated_alpha <- safe_solve(
        K_train, prior_all[train] + simulated_noise, case$cfg$jitter
      )
      posterior_mean <- as.vector(F_all %*% fit$beta) +
        as.vector(K_all_train %*% observed_alpha)
      posterior_residual_draw <- prior_all -
        as.vector(K_all_train %*% simulated_alpha)
      draw[, h] <- posterior_mean +
        case$cfg$reference_draw_scale * posterior_residual_draw
    }
  })
  draw
}

construct_reference_environment <- function(case) {
  reference_case <- case
  reference_case$cfg$kernel_gamma <- case$cfg$reference_kernel_gamma
  reference_model <- fit_gpfr_mle(
    reference_case$initial_Y, reference_case$initial_idx, reference_case,
    warm_model = NULL,
    role = "reference"
  )
  scores <- joint_reference_draw(reference_model, reference_case)
  curves <- scores %*% t(reference_model$coreg$curve_loading)
  J <- as.vector(scores %*% reference_model$coreg$objective_loading)
  case$reference <- list(
    model = reference_model, scores = scores, curves = curves, J = J,
    best_idx = which.min(J), generator_noise_sd = reference_model$sigma_e,
    construction = paste0(
      "one fixed method-neutral conditional GPFR reference path with gamma_u=",
      case$cfg$reference_kernel_gamma,
      " and gamma_x=", case$cfg$functional_kernel_gamma,
      "; prespecified seed; no pseudo-response refit or method-generated truth"
    )
  )
  case
}

measurement_noise_curve <- function(case, idx, visit) {
  with_seed(case$cfg$seed + idx * 1009L + visit * 97L, {
    rnorm(length(case$t), 0, case$reference$generator_noise_sd)
  })
}

observe_reference_candidate <- function(case, idx, visit) {
  as.vector(case$reference$curves[idx, ] +
              measurement_noise_curve(case, idx, visit))
}

# ==============================================================================
# 6. Scalar-loss FEI, local attenuation, full-pool D_M, EQI, and WDEI
# ==============================================================================

objective_noise_variance <- function(model) {
  # Propagate fitted observation noise from independent response-score space to
  # the scalar loss J. This is the discrete linear-loss analogue of q' Sigma_e q.
  g <- model$coreg$objective_loading
  score_noise <- vapply(model$score_fits, `[[`, numeric(1), "noise")
  as.numeric(sum(g^2 * score_noise))
}

compute_weight <- function(idx_history, case, pred, model) {
  counts <- tabulate(idx_history, nbins = nrow(case$U))
  evaluated <- which(counts > 0L)
  sigma_J_noise2 <- objective_noise_variance(model)
  tau_J <- case$cfg$kappa * sigma_J_noise2
  active <- evaluated[pred$varJ[evaluated] <= tau_J]
  logW <- numeric(nrow(case$U))
  if (length(active)) {
    for (idx in active) {
      d2 <- rowSums((case$U_unit - matrix(
        case$U_unit[idx, ], nrow(case$U), ncol(case$U_unit), TRUE
      ))^2)
      # Only uncertainty-activated historical evaluations contribute. At their
      # exact input the factor is rho; it approaches one with distance. Repeated
      # visits compound over the complete indexed evaluation history.
      local <- 1 - (1 - case$cfg$rho) * exp(-case$cfg$eta * d2)
      logW <- logW + counts[idx] * log(pmax(local, case$cfg$epsilon))
    }
  }
  list(W = exp(logW), active = active,
       active_unique = length(active),
       active_visits = if (length(active)) sum(counts[active]) else 0L,
       sigma_J_noise2 = sigma_J_noise2, tau_J = tau_J)
}

scalar_loss_fei <- function(pred, evaluated) {
  incumbent <- min(pred$muJ[evaluated])
  gaussian_ei(pred$muJ, pred$varJ, incumbent)
}

global_learning_gain <- function(model, pred) {
  # Exact D_M on the complete candidate library. For a prospective noisy curve
  # at candidate j, each score component contributes its full-pool posterior
  # covariance squared divided by latent variance plus future measurement noise.
  case <- model$case
  all_idx <- seq_len(nrow(case$U))
  batches <- split(all_idx, ceiling(seq_along(all_idx) /
                                      case$cfg$diversity_batch))
  D <- numeric(length(all_idx))
  F_all <- gpfr_feature_map(case, all_idx)
  for (h in seq_along(model$score_fits)) {
    fit <- model$score_fits[[h]]
    R_all_train <- candidate_kernel(case, all_idx, model$idx_obs,
                                    model$ell_u, model$ell_x)
    K_all_train <- fit$signal * R_all_train
    M_all <- K_all_train %*% fit$Kinv
    trend_all <- F_all - M_all %*% fit$F_train
    precomputed <- list(K_all_train = K_all_train, M_all = M_all,
                        trend_all = trend_all)
    for (batch in batches) {
      cross <- posterior_crosscov_block(model, h, all_idx, batch,
                                        precomputed)
      reduction <- colMeans(cross^2) /
        pmax(pred$var_score[batch, h] + fit$noise, 1e-14)
      D[batch] <- D[batch] +
        model$coreg$integrated_weights[h] * pmax(reduction, 0)
    }
  }
  pmax(D, 0)
}

expected_quantile_improvement <- function(pred, evaluated, model) {
  zq <- qnorm(model$case$cfg$eqi_quantile)
  current_q <- pred$muJ + zq * sqrt(pred$varJ)
  incumbent_q <- min(current_q[evaluated])
  g <- model$coreg$objective_loading
  noise <- vapply(model$score_fits, `[[`, numeric(1), "noise")
  mean_update_var <- rowSums(sweep(
    pred$var_score^2 /
      sweep(pred$var_score, 2, noise, "+"),
    2, g^2, "*"
  ))
  v_after <- pmax(pred$varJ - mean_update_var, 0)
  q_after_mean <- pred$muJ + zq * sqrt(v_after)
  gaussian_ei(q_after_mean, pmax(mean_update_var, 0), incumbent_q)
}

acquisition_values <- function(method, model, pred, idx_history) {
  case <- model$case
  evaluated <- unique(idx_history)
  FEI <- scalar_loss_fei(pred, evaluated)
  decay <- compute_weight(idx_history, case, pred, model)
  if (method == "WDEI") {
    D <- global_learning_gain(model, pred)
    weighted <- decay$W * FEI
    cI <- max(weighted) + case$cfg$epsilon
    cD <- max(D) + case$cfg$epsilon
    # In the revised manuscript alpha weights GLOBAL profile learning.
    score <- (1 - case$cfg$alpha) * weighted / cI +
      case$cfg$alpha * D / cD
  } else if (method == "GPFR-EQI") {
    D <- rep(NA_real_, length(FEI))
    weighted <- rep(NA_real_, length(FEI))
    cI <- cD <- NA_real_
    score <- expected_quantile_improvement(pred, evaluated, model)
  } else {
    stop("Unknown sequential method: ", method, call. = FALSE)
  }
  list(score = score, FEI = FEI, weighted_FEI = weighted,
       W = decay$W, D = D, cI = cI, cD = cD,
       active_decay_indices = decay$active,
       active_decay_unique = decay$active_unique,
       active_decay_visits = decay$active_visits,
       objective_noise_variance = decay$sigma_J_noise2,
       tau_J = decay$tau_J)
}

# ==============================================================================
# 7. Four methods under a common initial design and frozen reference world
# ==============================================================================

run_initial <- function(case) {
  model <- fit_gpfr_mle(case$initial_Y, case$initial_idx, case)
  list(method = "Initial LHS", idx_history = case$initial_idx,
       Y_obs = case$initial_Y, model = model,
       pred = predict_gpfr(model), history = NULL)
}

make_one_shot_indices <- function(case) {
  n_add <- case$cfg$N_total - case$cfg$M0
  targets <- conditional_lhs_targets(
    case$U_unit[case$initial_idx, , drop = FALSE], n_add,
    case$cfg$one_shot_restarts, case$cfg$seed + 3001L
  )
  add <- map_targets_to_unique_pool(
    targets, case$U_unit, case$initial_idx, case$cfg$seed + 3011L
  )
  list(add = add, targets = targets,
       conditional_min_distance = attr(targets, "conditional_min_distance"))
}

run_one_shot <- function(case) {
  design <- make_one_shot_indices(case)
  add <- design$add
  Yadd <- t(vapply(add, function(j) observe_reference_candidate(case, j, 1L),
                   numeric(length(case$t))))
  idx_history <- c(case$initial_idx, add)
  Y_obs <- rbind(case$initial_Y, Yadd)
  model <- fit_gpfr_mle(Y_obs, idx_history, case)
  list(method = "One-shot LHS", idx_history = idx_history,
       Y_obs = Y_obs, model = model, pred = predict_gpfr(model),
       history = NULL, design = design)
}

reference_pog <- function(case, idx) {
  # Revised-manuscript POG: reference functional-loss difference. Because the
  # candidate set is finite, exact zero is valid when the recommended evaluated
  # input is the reference-best candidate; no artificial curve-distance term is
  # added merely to prevent a zero.
  pmax(case$reference$J[idx] - min(case$reference$J), 0)
}

run_sequential <- function(method, case) {
  idx_history <- case$initial_idx
  Y_obs <- case$initial_Y
  n_add <- case$cfg$N_total - case$cfg$M0
  history <- data.frame(
    Iteration = seq_len(n_add), Method = method,
    Selected_Index = NA_integer_, Selected_Score = NA_real_,
    Selected_FEI = NA_real_, Selected_Weighted_FEI = NA_real_,
    Selected_Weight = NA_real_, Selected_D = NA_real_,
    D_CV = NA_real_, FEI_D_Spearman = NA_real_,
    Best_New_Index = NA_integer_, Best_New_Score = NA_real_,
    Best_Replicate_Index = NA_integer_, Best_Replicate_Score = NA_real_,
    Replicate_to_New_Score_Ratio = NA_real_,
    Max_FEI = NA_real_, Max_Weighted_FEI = NA_real_,
    Max_New_FEI = NA_real_, Max_Replicate_FEI = NA_real_,
    Max_New_D = NA_real_, Max_Replicate_D = NA_real_,
    Selected_Was_Replicate = NA,
    Selected_Replicate_Class = NA_character_,
    Cumulative_Replicates = NA_integer_,
    Useful_Replicate = NA, Redundant_Replicate = NA,
    Cumulative_Useful_Replicates = NA_integer_,
    Cumulative_Redundant_Replicates = NA_integer_,
    Active_Decay_Unique = NA_integer_, Active_Decay_Visits = NA_integer_,
    Objective_Noise_Variance = NA_real_, Tau_J = NA_real_,
    Selected_VarJ = NA_real_, Selected_VarJ_to_TauJ = NA_real_,
    Selected_Integrated_Variance = NA_real_,
    Recommendation_Index = NA_integer_, Reference_POG = NA_real_,
    Integrated_Latent_Variance = NA_real_,
    MLE_NLL = NA_real_, MLE_Convergence = NA_integer_,
    MLE_At_Boundary = NA, Ell_U = NA_real_, Ell_X = NA_real_,
    Measurement_SD = NA_real_, stringsAsFactors = FALSE
  )
  model <- fit_gpfr_mle(Y_obs, idx_history, case)
  pred <- predict_gpfr(model)
  for (s in seq_len(n_add)) {
    acq <- acquisition_values(method, model, pred, idx_history)
    evaluated_before <- unique(idx_history)
    new_before <- setdiff(seq_len(nrow(case$U)), evaluated_before)
    best_rep <- evaluated_before[which.max(acq$score[evaluated_before])]
    best_new <- new_before[which.max(acq$score[new_before])]
    selected <- which.max(acq$score)
    was_rep <- selected %in% idx_history
    useful_rep <- was_rep && pred$varJ[selected] > acq$tau_J
    redundant_rep <- was_rep && !useful_rep
    rep_class <- if (!was_rep) {
      "new"
    } else if (useful_rep) {
      "useful exact replication"
    } else {
      "redundant exact replication"
    }
    visit <- sum(idx_history == selected) + 1L
    ynew <- observe_reference_candidate(case, selected, visit)

    history$Selected_Index[s] <- selected
    history$Selected_Score[s] <- acq$score[selected]
    history$Selected_FEI[s] <- acq$FEI[selected]
    history$Selected_Weighted_FEI[s] <- if (method == "WDEI")
      acq$weighted_FEI[selected] else NA_real_
    history$Selected_Weight[s] <- acq$W[selected]
    history$Selected_D[s] <- if (method == "WDEI") acq$D[selected] else NA_real_
    history$D_CV[s] <- if (method == "WDEI")
      sd(acq$D) / max(mean(acq$D), case$cfg$epsilon) else NA_real_
    history$FEI_D_Spearman[s] <- if (method == "WDEI")
      suppressWarnings(cor(acq$FEI, acq$D, method = "spearman")) else NA_real_
    history$Best_New_Index[s] <- best_new
    history$Best_New_Score[s] <- acq$score[best_new]
    history$Best_Replicate_Index[s] <- best_rep
    history$Best_Replicate_Score[s] <- acq$score[best_rep]
    history$Replicate_to_New_Score_Ratio[s] <- acq$score[best_rep] /
      max(acq$score[best_new], case$cfg$epsilon)
    history$Max_FEI[s] <- max(acq$FEI)
    history$Max_Weighted_FEI[s] <- if (method == "WDEI")
      max(acq$weighted_FEI) else NA_real_
    history$Max_New_FEI[s] <- max(acq$FEI[new_before])
    history$Max_Replicate_FEI[s] <- max(acq$FEI[evaluated_before])
    history$Max_New_D[s] <- if (method == "WDEI")
      max(acq$D[new_before]) else NA_real_
    history$Max_Replicate_D[s] <- if (method == "WDEI")
      max(acq$D[evaluated_before]) else NA_real_
    history$Selected_Was_Replicate[s] <- was_rep
    history$Selected_Replicate_Class[s] <- rep_class
    history$Useful_Replicate[s] <- useful_rep
    history$Redundant_Replicate[s] <- redundant_rep
    history$Active_Decay_Unique[s] <- acq$active_decay_unique
    history$Active_Decay_Visits[s] <- acq$active_decay_visits
    history$Objective_Noise_Variance[s] <- acq$objective_noise_variance
    history$Tau_J[s] <- acq$tau_J
    history$Selected_VarJ[s] <- pred$varJ[selected]
    history$Selected_VarJ_to_TauJ[s] <- pred$varJ[selected] /
      max(acq$tau_J, case$cfg$epsilon)
    history$Selected_Integrated_Variance[s] <- pred$integrated[selected]

    idx_history <- c(idx_history, selected)
    Y_obs <- rbind(Y_obs, ynew)
    history$Cumulative_Replicates[s] <-
      length(idx_history) - length(unique(idx_history))
    history$Cumulative_Useful_Replicates[s] <-
      sum(history$Useful_Replicate[seq_len(s)])
    history$Cumulative_Redundant_Replicates[s] <-
      sum(history$Redundant_Replicate[seq_len(s)])

    # Theory: re-estimate GPFR before the next iteration. The old model supplies
    # only the numerical starting values; it does not freeze any hyperparameter.
    model <- fit_gpfr_mle(Y_obs, idx_history, case, warm_model = model)
    pred <- predict_gpfr(model)
    evaluated <- unique(idx_history)
    rec <- evaluated[which.min(pred$muJ[evaluated])]
    history$Recommendation_Index[s] <- rec
    history$Reference_POG[s] <- reference_pog(case, rec)
    history$Integrated_Latent_Variance[s] <- mean(pred$integrated)
    history$MLE_NLL[s] <- model$nll
    history$MLE_Convergence[s] <- model$convergence
    history$MLE_At_Boundary[s] <- model$parameter_at_boundary
    history$Ell_U[s] <- model$ell_u
    history$Ell_X[s] <- model$ell_x
    history$Measurement_SD[s] <- model$sigma_e
    if (case$cfg$verbose) cat(sprintf(
      "%s | iteration %02d/%02d | selected %d | %s | ell_u=%.3f ell_x=%.3f\n",
      method, s, n_add, selected, rep_class, model$ell_u, model$ell_x
    ))
  }
  list(method = method, idx_history = idx_history, Y_obs = Y_obs,
       model = model, pred = pred, history = history)
}

# ==============================================================================
# 8. Metrics and diagnostics
# ==============================================================================

nearest_history_distance <- function(idx_history, case) {
  if (length(idx_history) <= case$cfg$M0) return(numeric())
  out <- numeric(length(idx_history) - case$cfg$M0)
  for (k in (case$cfg$M0 + 1L):length(idx_history)) {
    prior <- idx_history[seq_len(k - 1L)]
    d <- sqrt(rowSums((case$U_unit[prior, , drop = FALSE] -
      matrix(case$U_unit[idx_history[k], ], length(prior), 3, TRUE))^2))
    out[k - case$cfg$M0] <- min(d)
  }
  out
}

global_r2 <- function(actual, predicted) {
  sse <- sum((actual - predicted)^2)
  sst <- sum((actual - mean(actual))^2)
  1 - sse / max(sst, 1e-14)
}

evaluate_method <- function(run, case) {
  evaluated <- unique(run$idx_history)
  rec <- evaluated[which.min(run$pred$muJ[evaluated])]
  error <- run$pred$mu_curve - case$reference$curves
  # Manuscript definition: average of the candidate-wise curve RMSE values.
  curve_rmse <- sqrt(rowSums(error^2 *
    matrix(case$basis$q, nrow(error), length(case$basis$q), TRUE)))
  global_rmse <- mean(curve_rmse)
  rec_error <- run$pred$mu_curve[rec, ] - case$reference$curves[rec, ]
  curve_gap <- case$reference$curves[rec, ] -
    case$reference$curves[case$reference$best_idx, ]
  exact_rep <- length(run$idx_history) - length(evaluated)
  near_d <- nearest_history_distance(run$idx_history, case)
  near_rep <- if (length(near_d)) sum(near_d > 0 &
    near_d <= case$cfg$near_replication_fraction * case$cfg$r0) else 0L
  useful_rep <- if (is.null(run$history)) NA_integer_ else
    sum(run$history$Useful_Replicate)
  redundant_rep <- if (is.null(run$history)) NA_integer_ else
    sum(run$history$Redundant_Replicate)
  activation_rate <- if (is.null(run$history) || run$method != "WDEI") {
    NA_real_
  } else {
    mean(run$history$Active_Decay_Unique > 0L)
  }
  data.frame(
    Method = run$method,
    Global_Reference_RMSE = global_rmse,
    Global_Reference_R2 = global_r2(case$reference$curves,
                                    run$pred$mu_curve),
    Integrated_Latent_Variance = mean(run$pred$integrated),
    Reference_POG = reference_pog(case, rec),
    Optimal_RMSE = sqrt(sum(case$basis$q * rec_error^2)),
    Reference_Best_Curve_Gap_RMSE = sqrt(sum(case$basis$q * curve_gap^2)),
    Recommended_Index = rec, Unique_Inputs = length(evaluated),
    Exact_Replicates = exact_rep,
    Useful_Replicates = useful_rep,
    Redundant_Replicates = redundant_rep,
    Activation_Rate = activation_rate,
    Near_Replicates = near_rep,
    Ell_U = run$model$ell_u,
    Ell_X = run$model$ell_x,
    Identifiable_Ell_U_Lower = case$cfg$ell_u_bounds[1],
    Identifiable_Ell_X_Lower = case$cfg$ell_x_bounds[1],
    Ell_At_Boundary = run$model$ell_at_boundary,
    Ell_X_At_Boundary = run$model$ell_x_at_boundary,
    Measurement_SD = run$model$sigma_e,
    MLE_NLL = run$model$nll,
    MLE_Convergence = run$model$convergence,
    Noise_At_Boundary = run$model$noise_at_boundary,
    Signal_At_Boundary = run$model$signal_at_boundary,
    MLE_At_Boundary = run$model$parameter_at_boundary,
    stringsAsFactors = FALSE
  )
}

recommendation_table <- function(runs, case) {
  do.call(rbind, lapply(runs, function(run) {
    evaluated <- unique(run$idx_history)
    rec <- evaluated[which.min(run$pred$muJ[evaluated])]
    data.frame(
      Method = run$method, Candidate_Index = rec,
      Layer_Thickness_mm = case$U[rec, 1],
      Infilling_Rate = case$U[rec, 2],
      Printing_Speed_mm_s = case$U[rec, 3],
      Posterior_Mean_J = run$pred$muJ[rec],
      Reference_J = case$reference$J[rec],
      Reference_POG = reference_pog(case, rec),
      stringsAsFactors = FALSE
    )
  }))
}

mechanism_table <- function(runs) {
  do.call(rbind, lapply(runs[c("GPFR-EQI", "WDEI")], function(run) {
    h <- run$history
    data.frame(
      Method = run$method,
      Exact_Replicates = sum(h$Selected_Was_Replicate),
      Useful_Replicates = sum(h$Useful_Replicate),
      Redundant_Replicates = sum(h$Redundant_Replicate),
      Mean_Active_Decay_Unique = if (run$method == "WDEI")
        mean(h$Active_Decay_Unique) else NA_real_,
      Activation_Rate = if (run$method == "WDEI")
        mean(h$Active_Decay_Unique > 0L) else NA_real_,
      Minimum_Selected_Weight = if (run$method == "WDEI")
        min(h$Selected_Weight) else NA_real_,
      Mean_Selected_Weight = if (run$method == "WDEI")
        mean(h$Selected_Weight) else NA_real_,
      Final_Tau_J = tail(h$Tau_J, 1),
      Rounds_Selected_Weight_Below_One = if (run$method == "WDEI")
        sum(h$Selected_Weight < 1 - 1e-10) else NA_integer_,
      Mean_D_CV = if (run$method == "WDEI") mean(h$D_CV) else NA_real_,
      Mean_FEI_D_Spearman = if (run$method == "WDEI")
        mean(h$FEI_D_Spearman, na.rm = TRUE) else NA_real_,
      Mean_Replicate_to_New_Score_Ratio =
        mean(h$Replicate_to_New_Score_Ratio),
      Max_Replicate_to_New_Score_Ratio =
        max(h$Replicate_to_New_Score_Ratio),
      Rounds_Replicate_Within_5pct_of_New =
        sum(h$Replicate_to_New_Score_Ratio >= 0.95),
      Final_Integrated_Latent_Variance = tail(h$Integrated_Latent_Variance, 1),
      Final_Reference_POG = tail(h$Reference_POG, 1),
      MLE_Boundary_Rounds = sum(h$MLE_At_Boundary),
      stringsAsFactors = FALSE
    )
  }))
}

# ==============================================================================
# 9. Plots: active R/RStudio Plots device only
# ==============================================================================

plot_recommended_curves <- function(result) {
  old <- par(no.readonly = TRUE); on.exit(par(old), add = TRUE)
  par(family = "serif", mfrow = c(2, 2),
      mar = c(3.4, 3.6, 2.8, 0.6), oma = c(0.2, 0.2, 0.2, 0.1),
      mgp = c(2.15, 0.65, 0), tcl = -0.22, las = 1,
      cex.axis = 0.80, cex.lab = 0.88)
  panel_labels <- c("(a)", "(b)", "(c)", "(d)")
  optimal <- result$case$reference$curves[result$case$reference$best_idx, ]
  for (i in seq_along(result$runs)) {
    run <- result$runs[[i]]
    evaluated <- unique(run$idx_history)
    rec <- evaluated[which.min(run$pred$muJ[evaluated])]
    mu <- run$pred$mu_curve[rec, ]
    sdv <- sqrt(pmax(run$pred$var_curve[rec, ], 0))
    true_at_rec <- result$case$reference$curves[rec, ]
    lower <- mu - 1.96 * sdv
    upper <- mu + 1.96 * sdv
    ylim <- range(lower, upper, true_at_rec, optimal, finite = TRUE)
    subtitle <- sprintf("u=(%.3f, %.3f, %.2f)",
      result$case$U[rec, 1], result$case$U[rec, 2], result$case$U[rec, 3])
    plot(result$case$t, mu, type = "n", ylim = ylim,
         xlab = "Time (s)", ylab = "Functional response",
         main = paste0(run$method, "\n", subtitle), font.main = 2,
         cex.main = 0.86)
    grid(col = "#D9D9D9", lty = 3)
    polygon(c(result$case$t, rev(result$case$t)), c(lower, rev(upper)),
            col = "#E3E3E3", border = NA)
    lines(result$case$t, optimal, col = "black", lty = 3, lwd = 2.0)
    lines(result$case$t, true_at_rec, col = "#0072B2", lty = 1, lwd = 2.0)
    lines(result$case$t, mu, col = "#D55E00", lty = 2, lwd = 2.0)
    box()
    legend("topleft",
           c("Reference-best feasible curve", "Reference curve at recommendation",
             "GPFR posterior mean", "95% latent-mean interval"),
           col = c("black", "#0072B2", "#D55E00", "#E3E3E3"),
           lty = c(3, 1, 2, NA), lwd = c(2, 2, 2, NA),
           pch = c(NA, NA, NA, 15), pt.cex = 1.45,
           bty = "n", cex = 0.57, y.intersp = 0.88)
    mtext(panel_labels[i], side = 3, line = 1.35, adj = 0,
          font = 2, cex = 0.90)
  }
}

plot_performance <- function(result) {
  old <- par(no.readonly = TRUE); on.exit(par(old), add = TRUE)
  par(family = "serif", mfrow = c(2, 2), mar = c(4.5, 3.8, 2.2, 0.5),
      mgp = c(2.25, 0.65, 0), tcl = -0.22, las = 1,
      cex.axis = 0.78, cex.lab = 0.86, cex.main = 0.86)
  metrics <- c("Global_Reference_RMSE", "Integrated_Latent_Variance",
               "Reference_POG", "Optimal_RMSE")
  labels <- c("Global reference RMSE", "Integrated latent variance",
              "Reference POG (loss gap)", "Optimal RMSE")
  panel <- c("(a)", "(b)", "(c)", "(d)")
  for (i in seq_along(metrics)) {
    barplot(result$results[[metrics[i]]],
            names.arg = c("Initial", "One-shot", "EQI", "WDEI"),
            col = c("#666666", "#E69F00", "#2CA02C", "#1F77B4"),
            border = NA, ylab = labels[i], las = 2,
            main = paste(panel[i], labels[i]))
    box(bty = "l")
  }
}

plot_wdei_mechanism <- function(result) {
  hw <- result$runs$WDEI$history
  he <- result$runs$`GPFR-EQI`$history
  old <- par(no.readonly = TRUE); on.exit(par(old), add = TRUE)
  par(family = "serif", mfrow = c(2, 2),
      mar = c(3.4, 3.8, 2.5, 0.6), oma = c(0.2, 0.2, 0.2, 0.1),
      mgp = c(2.25, 0.65, 0), tcl = -0.22, las = 1,
      cex.axis = 0.80, cex.lab = 0.88, cex.main = 0.88)

  ymax_active <- max(c(hw$Active_Decay_Unique, 1), na.rm = TRUE)
  plot(hw$Iteration, hw$Active_Decay_Unique, type = "o", pch = 19,
       col = "#D62728", xlab = "Sequential iteration",
       ylim = c(0, ymax_active), ylab = "Activated decay locations",
       main = "(a) Uncertainty-activated decay")
  grid(col = "#D9D9D9", lty = 3)
  lines(hw$Iteration, hw$Active_Decay_Unique, type = "o", pch = 19,
        col = "#D62728")

  ymax_rep <- max(c(hw$Cumulative_Redundant_Replicates,
                    he$Cumulative_Redundant_Replicates, 1), na.rm = TRUE)
  plot(hw$Iteration, hw$Cumulative_Redundant_Replicates, type = "n",
       xlab = "Sequential iteration", ylim = c(0, ymax_rep),
       ylab = "Cumulative redundant replicates",
       main = "(b) Redundant replication")
  grid(col = "#D9D9D9", lty = 3)
  lines(hw$Iteration, hw$Cumulative_Redundant_Replicates,
        type = "o", pch = 19,
        col = "#1F77B4")
  lines(he$Iteration, he$Cumulative_Redundant_Replicates,
        type = "o", pch = 17, col = "#2CA02C")
  legend("topleft", c("WDEI", "GPFR-EQI"),
         col = c("#1F77B4", "#2CA02C"), pch = c(19, 17),
         lty = 1, bty = "n")

  plot(hw$Iteration, hw$Max_FEI, type = "o", pch = 19,
       col = "#444444", xlab = "Sequential iteration",
       ylab = "Maximum FEI of J",
       main = "(c) Original and attenuated FEI")
  grid(col = "#D9D9D9", lty = 3)
  lines(hw$Iteration, hw$Max_FEI, type = "o", pch = 19,
        col = "#444444")
  lines(hw$Iteration, hw$Max_Weighted_FEI, type = "o", pch = 17,
        col = "#D62728")
  legend("topright", c("Original FEI", "Weighted FEI"),
         col = c("#444444", "#D62728"), pch = c(19, 17),
         lty = 1, bty = "n")

  plot(hw$Iteration, hw$Integrated_Latent_Variance, type = "o", pch = 19,
       col = "#1F77B4", xlab = "Sequential iteration",
       ylab = "Integrated latent variance",
       main = "(d) Global uncertainty reduction")
  grid(col = "#D9D9D9", lty = 3)
  lines(hw$Iteration, hw$Integrated_Latent_Variance, type = "o", pch = 19,
        col = "#1F77B4")
  lines(he$Iteration, he$Integrated_Latent_Variance, type = "o", pch = 17,
        col = "#2CA02C")
  legend("topright", c("WDEI", "GPFR-EQI"),
         col = c("#1F77B4", "#2CA02C"), pch = c(19, 17),
         lty = 1, bty = "n")
}

show_case16_figure <- function(number, result = Case3D16_Result) {
  number <- as.integer(number)
  assert_true(number %in% 1:3, "Figure number must be 1, 2, or 3.")
  switch(as.character(number),
         `1` = plot_recommended_curves(result),
         `2` = plot_performance(result),
         `3` = plot_wdei_mechanism(result))
  invisible(NULL)
}

activate_rstudio_plots_device <- function() {
  # Identical device strategy to 3D case 12. RStudioGD is the graphics device
  # behind the lower-right Plots pane. This does not open an external window.
  if (identical(Sys.getenv("RSTUDIO"), "1")) {
    options(device = "RStudioGD")
    devices <- grDevices::dev.list()
    hit <- if (is.null(devices)) integer() else
      which(names(devices) == "RStudioGD")
    if (length(hit)) {
      grDevices::dev.set(devices[hit[length(hit)]])
    } else {
      grDevices::dev.new(noRStudioGD = FALSE)
    }
  }
  invisible(grDevices::dev.cur())
}

display_case16_figures <- function(result) {
  activate_rstudio_plots_device()
  # Explicit high-level calls create three pages in RStudio Plot History.
  plot_performance(result)
  plot_wdei_mechanism(result)
  plot_recommended_curves(result)
  if (identical(Sys.getenv("RSTUDIO"), "1") &&
      requireNamespace("rstudioapi", quietly = TRUE) &&
      rstudioapi::isAvailable()) {
    try(rstudioapi::executeCommand("activatePlots"), silent = TRUE)
  }
  cat("\nFigures were drawn directly in RStudio > Plots.\n",
      "The recommendation-curve figure is the visible final page; ",
      "use the Plots back arrow for the other two pages.\n", sep = "")
  invisible(TRUE)
}

# ==============================================================================
# 10. Outputs and main program
# ==============================================================================

write_outputs <- function(result) {
  out <- result$case$cfg$output_dir
  dir.create(out, recursive = TRUE, showWarnings = FALSE)
  write.csv(result$results, file.path(out, "Table_3D_case16_results.csv"),
            row.names = FALSE)
  write.csv(result$recommendations,
            file.path(out, "Table_3D_case16_recommendations.csv"),
            row.names = FALSE)
  write.csv(result$mechanism,
            file.path(out, "Table_3D_case16_mechanism.csv"),
            row.names = FALSE)
  history <- do.call(rbind, lapply(
    result$runs[c("GPFR-EQI", "WDEI")], `[[`, "history"
  ))
  write.csv(history, file.path(out, "Sequential_history_3D_case16.csv"),
            row.names = FALSE)
  candidate_library <- data.frame(
    Candidate_Index = seq_len(nrow(result$case$U)), result$case$U,
    Functional_Covariate_Source = c(
      rep(result$case$data$functional_source_initial, result$case$cfg$M0),
      rep(if (isTRUE(result$case$data$functional_observed))
            "Fixed Triangular(202,205,208) planned candidate trajectory" else
            "Unavailable; omitted from GPFR regression",
          nrow(result$case$U) - result$case$cfg$M0)
    ), result$case$X, check.names = FALSE
  )
  names(candidate_library)[6:ncol(candidate_library)] <-
    paste0("Functional_t", seq_len(ncol(result$case$X)))
  write.csv(candidate_library,
            file.path(out, "Candidate_library_3D_case16.csv"),
            row.names = FALSE)

  compact <- result
  compact$case$Du_all <- NULL
  compact$case$Dx_all <- NULL
  compact$case$reference$model$case$Du_all <- NULL
  compact$case$reference$model$case$Dx_all <- NULL
  compact$runs <- lapply(compact$runs, function(run) {
    run$model$case$Du_all <- NULL
    run$model$case$Dx_all <- NULL
    run$model$case$reference$model$case$Du_all <- NULL
    run$model$case$reference$model$case$Dx_all <- NULL
    run
  })
  saveRDS(compact, file.path(out, "3D_case16_result.rds"))

  manifest <- c(
    "3D case 16 - theory-aligned semi-synthetic sequential benchmark",
    paste("Seed:", result$case$cfg$seed),
    paste("Physical initial curves:", result$case$cfg$M0),
    paste("Total curve budget:", result$case$cfg$N_total),
    paste("Candidate size:", result$case$cfg$candidate_size),
    paste("Reference construction:", result$case$reference$construction),
    paste("Reference random features:", result$case$cfg$reference_rff),
    paste("Reference kernel gamma:", result$case$cfg$reference_kernel_gamma),
    paste("Functional-input PCs:", ncol(result$case$X_scores)),
    paste("Response score components:",
          result$case$coreg_template$response_components),
    paste("Response variance explained:", signif(
      result$case$coreg_template$response_variance_explained, 6)),
    paste("Functional input observed:",
          isTRUE(result$case$data$functional_observed)),
    paste("Reference observation-noise SD:",
          signif(result$case$reference$generator_noise_sd, 6)),
    paste("Noise initialization source:", result$case$noise_start$source),
    paste("alpha, kappa, rho:", result$case$cfg$alpha,
          result$case$cfg$kappa, result$case$cfg$rho),
    paste("eta, r0:", signif(result$case$cfg$eta, 6),
          signif(result$case$cfg$r0, 6)),
    paste("Requested ell_u bounds:", paste(
      signif(result$case$cfg$ell_u_requested_bounds, 6), collapse = ", ")),
    paste("Identifiable ell_u bounds:", paste(
      signif(result$case$cfg$ell_u_bounds, 6), collapse = ", ")),
    paste("Requested ell_x bounds:", paste(
      signif(result$case$cfg$ell_x_requested_bounds, 6), collapse = ", ")),
    paste("Identifiable ell_x bounds:", paste(
      signif(result$case$cfg$ell_x_bounds, 6), collapse = ", ")),
    "GPFR kernel, signal, and iid measurement-noise parameters are refitted by MLE each round.",
    "The common response basis and functional-input covariance coordinates are fixed across methods.",
    "Scalar printing parameters enter the GPFR mean; supplied functional trajectories enter the covariance kernel.",
    "Legacy functional covariates are deterministic triangular protocol trajectories; supplied measured CSV trajectories take precedence.",
    "WDEI uses scalar-loss FEI and full-pool expected integrated latent variance reduction D_M.",
    "WDEI = (1-alpha)*normalized(W*FEI) + alpha*normalized(D_M); attenuation is applied only to FEI.",
    "Decay is activated only where posterior Var[J] <= tau_J = kappa*objective observation-noise variance.",
    "POG is reference J at the recommendation minus the best reference J over the candidate set.",
    "Reference truth uses no 200-point pseudo-response refit.",
    "One-shot additions use an independent randomized conditional LHS.",
    "RMSE and POG are reference-relative, not unknown physical-truth errors.",
    "The final high-level base-graphics calls draw directly on RStudio's current Plots device."
  )
  writeLines(manifest, file.path(out, "Run_manifest_3D_case16.txt"))
}

print_results <- function(result) {
  cat("\n================ 3D case 16: main results ================\n")
  print(result$results, row.names = FALSE, digits = 6)
  cat("\n================ Recommendations ================\n")
  print(result$recommendations, row.names = FALSE, digits = 6)
  cat("\n================ WDEI mechanism ================\n")
  print(result$mechanism, row.names = FALSE, digits = 6)
  cat("\nReference and fitting diagnostics:\n")
  cat("- Physical training curves =", result$case$cfg$M0,
      "; candidate library =", result$case$cfg$candidate_size, "\n")
  cat("- Reference = one fixed, method-neutral conditional GPFR path;",
      "no 200-point calibration refit or method-generated truth.\n")
  cat("- Reference observation-noise SD =",
      signif(result$case$reference$generator_noise_sd, 6), "\n")
  cat("- Functional-input covariance PCs =", ncol(result$case$X_scores),
      "; fitted ell_x =",
      signif(result$runs[[1]]$model$ell_x, 6), ".\n")
  cat("- Common response-score components =",
      result$case$coreg_template$response_components,
      "; retained variance =",
      signif(result$case$coreg_template$response_variance_explained, 6), "\n")
  cat("- Functional input observed =",
      isTRUE(result$case$data$functional_observed),
      if (isTRUE(result$case$data$functional_observed))
        "; supplied/planned trajectories are used.\n" else
        "; the unavailable legacy predictor is omitted.\n")
  cat("- Initial-design resolution r0 =", signif(result$case$cfg$r0, 6),
      "; identifiable ell_u lower bound =",
      signif(result$case$cfg$ell_u_bounds[1], 6), "\n")
  cat("- Functional-kernel resolution r0_x =",
      signif(result$case$cfg$r0_x, 6),
      "; identifiable ell_x lower bound =",
      signif(result$case$cfg$ell_x_bounds[1], 6), "\n")
  cat("- WDEI acquisition = (1-alpha)*normalized(W*FEI) +",
      " alpha*normalized(D_M), where alpha weights global learning.\n")
  cat("- Local decay activates at Var[J] <= tau_J = kappa*sigma_J,noise^2;",
      " kappa =", result$case$cfg$kappa, ".\n")
  cat("- POG is the reference scalar functional-loss gap; exact zero is valid.\n")
  cat("- Global RMSE is the average of candidate-wise curve RMSE values.\n")
  cat("- Final recommendations are restricted to evaluated feasible inputs.\n")
  eqi_pog <- result$results$Reference_POG[
    result$results$Method == "GPFR-EQI"
  ]
  wdei_pog <- result$results$Reference_POG[
    result$results$Method == "WDEI"
  ]
  cat("- Observed WDEI minus GPFR-EQI POG =",
      signif(wdei_pog - eqi_pog, 6),
      "(a realized comparison, not a hard-coded ordering).\n")
  cat("- Smoke/diagnostic results use generated test data and are not expected",
      " numerical values for the supplied 15 physical curves.\n")
}

main <- function(cfg = CFG) {
  assert_true(cfg$M0 == 15L, "The 3D-printing case requires 15 physical initial runs.")
  assert_true(cfg$N_total >= cfg$M0, "N_total must be no smaller than M0.")
  assert_true(cfg$candidate_size >= cfg$N_total, "Candidate set is too small.")
  assert_true(cfg$kernel_gamma %in% c(1, 2),
              "kernel_gamma must be 1 or 2.")
  assert_true(cfg$functional_kernel_gamma %in% c(1, 2),
              "functional_kernel_gamma must be 1 or 2.")
  assert_true(cfg$reference_kernel_gamma %in% c(1, 2),
              "reference_kernel_gamma must be 1 or 2.")
  assert_true(is.finite(cfg$ell_resolution_fraction) &&
                cfg$ell_resolution_fraction > 0,
              "ell_resolution_fraction must be positive.")
  assert_true(cfg$alpha >= 0 && cfg$alpha <= 1,
              "alpha must be in [0, 1].")
  assert_true(is.finite(cfg$kappa) && cfg$kappa > 0,
              "kappa must be positive.")
  assert_true(cfg$rho > 0 && cfg$rho <= 1,
              "rho must be in (0, 1].")
  cat("Loading the 15 initial 3D-printing experiments...\n")
  data <- load_case_data(cfg)
  cat("Building one fixed candidate library...\n")
  case <- build_case(data, cfg)
  case$coreg_template <- initialize_coregionalization(case)
  cat("Constructing one fixed method-neutral conditional GPFR reference...\n")
  case <- construct_reference_environment(case)
  cat("Running Initial LHS and independent randomized One-shot LHS...\n")
  initial <- run_initial(case)
  one_shot <- run_one_shot(case)
  cat("Running GPFR-EQI with round-wise GPFR MLE...\n")
  eqi <- run_sequential("GPFR-EQI", case)
  cat("Running WDEI with scalar-loss FEI, full-pool D_M, and round-wise GPFR MLE...\n")
  wdei <- run_sequential("WDEI", case)
  runs <- list("Initial LHS" = initial, "One-shot LHS" = one_shot,
               "GPFR-EQI" = eqi, "WDEI" = wdei)
  results <- do.call(rbind, lapply(runs, evaluate_method, case = case))
  rownames(results) <- NULL
  out <- list(case = case, runs = runs, results = results,
              recommendations = recommendation_table(runs, case),
              mechanism = mechanism_table(runs))
  print_results(out)
  if (cfg$save_outputs) write_outputs(out)
  invisible(out)
}

if (identical(Sys.getenv("CASE3D_SKIP_MAIN", unset = "0"), "1")) {
  Case3D16_Result <- NULL
} else {
  Case3D16_Result <- main(CFG)
  # Final executable plotting section copied from 3D case 12. Running the whole
  # file in RStudio always reaches these calls after calculation and file output.
  if (isTRUE(CFG$show_plots)) display_case16_figures(Case3D16_Result)
}
