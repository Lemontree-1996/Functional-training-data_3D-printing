# ==============================================================================
# Section4.R
# Complete R code for the two-dimensional linear example in Section 4

rm(list = ls())
options(stringsAsFactors = FALSE, scipen = 6, digits = 5)

# ==============================================================================
# 0. Run settings
# ==============================================================================

RUN_MAIN <- TRUE
RUN_ABLATION <- TRUE
RUN_DESIGN_BUDGET <- TRUE
RUN_ALPHA_SENSITIVITY <- TRUE
RUN_REGULATION_SENSITIVITY <- TRUE
RUN_SCALABILITY <- TRUE
RUN_APPENDIX_6D <- FALSE

SAVE_PAPER_OUTPUTS <- TRUE
OUTPUT_DIR <- "IISE_4_outputs"

N_REP_MAIN <- 1
N_REP_ABLATION <- 1
N_REP_DESIGN_BUDGET <- 1
N_REP_SENSITIVITY <- 1
N_REP_SCALABILITY <- 1

# Main configuration for the Section 4 example.
CFG <- list(
  case_id = "section4",
  seed = 20260805,
  n_time = 41,                 # 41 equally spaced points on [-2, 2]
  t_min = -2,
  t_max = 2,
  grid_each = 41,              # 41 x 41 candidate grid
  candidate_size = 1681,
  input_lower = c(-2, 5),
  input_upper = c(2, 9),
  M_init = 20,
  N_total = 40,
  n_basis = 9,
  spline_degree = 3,
  basis_penalty = 1e-6,
  process_var = 2,
  process_range = 1,
  noise_sd = 1,                # Var[epsilon_m(t)] = 1
  functional_covariate_sd = 0.001, # x_m(t)=exp(t)+v_m, v_m~N(0,0.001^2)
  scenario = "linear",
  surrogate = "linear",
  loss_type = "target_l2",
  target_u = c(-0.20, 6.80),
  objective_mc = 256,          # Common random numbers for FEI evaluation
  alpha = 0.5,
  rho = 0.50,
  eta = NA_real_,
  eta_multiplier = 1.0,
  decay_at_r0 = 0.05,
  kappa = 1.0,                 # tau_J=kappa*sigma^2_{epsilon,J}
  eqi_beta = 0.90,
  diversity_reference = 1681,
  kernel_ell_u = 0.35,
  kernel_ell_x = 1.0,
  tune_hyper = TRUE,
  hyper_grid_u = c(0.20, 0.35, 0.60),
  hyper_grid_x = c(0.60, 1.00, 1.80),
  jitter = 1e-9,
  utility_epsilon = 1e-12,
  verbose = TRUE
)

# Use Times New Roman on Windows and the portable serif family elsewhere.
PLOT_FAMILY <- "serif"
PDF_FAMILY <- "serif"
if (.Platform$OS.type == "windows") {
  windowsFonts(TNR = windowsFont("Times New Roman"))
  PLOT_FAMILY <- "TNR"
  PDF_FAMILY <- "Times New Roman"
}

set_plot_style <- function(mfrow = c(1, 1), mar = c(4.2, 4.6, 2.0, 0.8),
                           oma = c(0, 0, 0, 0)) {
  par(
    family = PLOT_FAMILY, mfrow = mfrow, mar = mar, oma = oma,
    mgp = c(2.55, 0.72, 0), tcl = -0.25,
    cex.axis = 0.98, cex.lab = 1.08, cex.main = 1.08,
    font.lab = 1, font.main = 2, las = 1, bty = "o",
    xaxs = "r", yaxs = "r"
  )
}

MAIN_METHODS <- c("Initial LHS", "One-shot LHS", "GPFR-EQI", "WDEI")

ABLATION_METHODS <- c("GPFR-FEI", "FEI+W", "FEI+D", "WDEI")

# ==============================================================================
# 1. Numerical utilities
# ==============================================================================

with_local_seed <- function(seed, expr) {
  old_exists <- exists(".Random.seed", envir = .GlobalEnv, inherits = FALSE)
  if (old_exists) old_seed <- get(".Random.seed", envir = .GlobalEnv)
  on.exit({
    if (old_exists) {
      assign(".Random.seed", old_seed, envir = .GlobalEnv)
    } else if (exists(".Random.seed", envir = .GlobalEnv, inherits = FALSE)) {
      rm(".Random.seed", envir = .GlobalEnv)
    }
  }, add = TRUE)
  set.seed(as.integer(seed %% .Machine$integer.max))
  force(expr)
}

trapezoid_weights <- function(t) {
  n <- length(t)
  stopifnot(n >= 2L, all(diff(t) > 0))
  w <- numeric(n)
  w[1] <- (t[2] - t[1]) / 2
  w[n] <- (t[n] - t[n - 1]) / 2
  if (n > 2L) {
    w[2:(n - 1)] <- (t[3:n] - t[1:(n - 2)]) / 2
  }
  w
}

safe_chol <- function(K, jitter = 1e-9, max_try = 8L) {
  K <- (K + t(K)) / 2
  for (i in 0:max_try) {
    add <- jitter * 10^i
    ans <- try(chol(K + diag(add, nrow(K))), silent = TRUE)
    if (!inherits(ans, "try-error")) return(ans)
  }
  stop("Cholesky decomposition failed; check the kernel matrix, noise variance, and duplicated inputs.")
}

chol_solve <- function(U, b) {
  backsolve(U, forwardsolve(t(U), b))
}

lhs_unit <- function(n, d, seed = NULL) {
  draw <- function() {
    ans <- matrix(NA_real_, n, d)
    for (j in seq_len(d)) {
      ans[, j] <- (sample.int(n) - runif(n)) / n
    }
    ans
  }
  if (is.null(seed)) draw() else with_local_seed(seed, draw())
}

derive_eta_from_initial_design <- function(U_unit, initial_idx, cfg) {
  X <- U_unit[initial_idx, , drop = FALSE]
  if (nrow(X) < 2L) stop("M_init must be at least 2 to calibrate eta.")
  distance_matrix <- as.matrix(stats::dist(X))
  diag(distance_matrix) <- Inf
  r0 <- median(apply(distance_matrix, 1, min))
  if (!is.finite(r0) || r0 <= 0) {
    stop("The median nearest-neighbor distance is invalid; eta cannot be calibrated.")
  }
  eta0 <- -log(cfg$decay_at_r0) / r0^2
  list(r0 = r0, eta0 = eta0,
       eta = cfg$eta_multiplier * eta0)
}

match_lhs_to_candidates <- function(lhs, U_unit) {
  selected <- integer(0)
  for (i in seq_len(nrow(lhs))) {
    d2 <- rowSums((sweep(U_unit, 2, lhs[i, ], "-"))^2)
    d2[selected] <- Inf
    selected <- c(selected, which.min(d2))
  }
  selected
}

maximin_lhs_indices <- function(n, U_unit, n_starts = 100L, seed = NULL) {
  draw <- function() {
    best_idx <- NULL
    best_score <- -Inf
    for (s in seq_len(n_starts)) {
      lhs <- lhs_unit(n, ncol(U_unit))
      idx <- match_lhs_to_candidates(lhs, U_unit)
      dmat <- as.matrix(stats::dist(U_unit[idx, , drop = FALSE]))
      diag(dmat) <- Inf
      score <- min(dmat)
      if (score > best_score) {
        best_score <- score
        best_idx <- idx
      }
    }
    best_idx
  }
  if (is.null(seed)) draw() else with_local_seed(seed, draw())
}

expected_improvement <- function(mu, var, incumbent) {
  sd <- sqrt(pmax(var, 0))
  delta <- incumbent - mu
  ans <- numeric(length(mu))
  regular <- sd > 1e-12
  z <- delta[regular] / sd[regular]
  ans[regular] <- delta[regular] * pnorm(z) + sd[regular] * dnorm(z)
  ans[!regular] <- pmax(delta[!regular], 0)
  pmax(ans, 0)
}

summarize_metrics <- function(result_df, metrics) {
  methods <- unique(result_df$Method)
  out <- list()
  counter <- 1L
  for (method in methods) {
    sub <- result_df[result_df$Method == method, , drop = FALSE]
    for (metric in metrics) {
      out[[counter]] <- data.frame(
        Method = method, Metric = metric,
        Value = sub[[metric]][1],
        row.names = NULL
      )
      counter <- counter + 1L
    }
  }
  do.call(rbind, out)
}

summarize_by_setting <- function(df, setting, metrics,
                                 method_column = NULL) {
  settings <- unique(df[[setting]])
  methods <- if (is.null(method_column)) NA_character_ else
    unique(df[[method_column]])
  rows <- list()
  k <- 1L
  for (s in settings) {
    for (method in methods) {
      keep <- df[[setting]] == s
      if (!is.null(method_column)) {
        keep <- keep & df[[method_column]] == method
      }
      sub <- df[keep, , drop = FALSE]
      for (metric in metrics) {
        rows[[k]] <- data.frame(
          Setting = s,
          Method = if (is.null(method_column)) NA_character_ else method,
          Metric = metric,
          Value = sub[[metric]][1],
          row.names = NULL
        )
        k <- k + 1L
      }
    }
  }
  do.call(rbind, rows)
}

compact_setting_table <- function(df, setting,
                                  metrics = c("Global_RMSE", "Opt_RMSE",
                                              "POG")) {
  settings <- sort(unique(df[[setting]]))
  rows <- lapply(settings, function(s) {
    sub <- df[df[[setting]] == s, , drop = FALSE]
    out <- data.frame(Setting = s, check.names = FALSE)
    for (metric in metrics) {
      out[[metric]] <- sprintf("%.4f", sub[[metric]][1])
    }
    out
  })
  do.call(rbind, rows)
}

# ==============================================================================
# 2. Functional responses, engineering loss, and simulated data
# ==============================================================================

make_basis_objects <- function(t, cfg) {
  B <- splines::bs(
    t, df = cfg$n_basis, degree = cfg$spline_degree,
    intercept = TRUE, Boundary.knots = range(t)
  )
  H <- ncol(B)
  D2 <- diff(diag(H), differences = 2)
  penalty <- crossprod(D2)
  A <- solve(crossprod(B) + cfg$basis_penalty * penalty, t(B))
  
  trap <- trapezoid_weights(t)
  q_raw <- exp(t)
  loss_weights <- trap * q_raw
  loss_weights <- loss_weights / sum(loss_weights)
  
  loading <- as.vector(crossprod(B, loss_weights))
  loss_basis_cross <- crossprod(B, B * loss_weights)
  
  domain_weights <- trap / sum(trap)
  integrated_basis_cross <- crossprod(
    B, B * domain_weights
  )
  integrated_basis_sq <- diag(integrated_basis_cross)
  
  list(B = B, A = A, loading = loading,
       loss_basis_cross = loss_basis_cross,
       loss_weights = loss_weights,
       domain_weights = domain_weights,
       integrated_basis_sq = integrated_basis_sq,
       integrated_basis_cross = integrated_basis_cross)
}

true_mean_curve <- function(u, t, scenario = "linear") {
  u <- as.numeric(u)
  if (scenario == "linear") {
    stopifnot(length(u) == 2L)
    return(u[1] * cos(t^2) + u[2] * sin((0.5 * t)^3))
  }
  if (scenario != "appendix6d") {
    stop("Unknown data-generating scenario: ", scenario)
  }
  stopifnot(length(u) == 6L)
  
  phi <- rbind(
    0.5 * t,
    cos(pi * t),
    t^2,
    2 * t,
    exp(t)
  )
  f <- c(
    sum(u^2) - 0.8 * cos(2 * pi * u[1] * u[2]),
    exp(-u[3]) * sin(2 * pi * u[4]),
    (u[5] - 0.5)^2 + u[1] * u[6],
    1.2 * cos(pi * u[2]) * cos(pi * u[5]),
    sqrt(u[1] * u[3] + 0.01) + u[4]^2
  )
  as.vector(f %*% phi)
}

objective_from_coefficients <- function(C, sim) {
  if (is.vector(C)) C <- matrix(C, nrow = 1L)
  D <- sweep(C, 2, sim$target_coef, "-")
  rowSums((D %*% sim$basis$loss_basis_cross) * D)
}

objective_moments <- function(mu_coef, var_coef, sim) {
  Q <- sim$basis$loss_basis_cross
  D <- sweep(mu_coef, 2, sim$target_coef, "-")
  mean_J <- rowSums((D %*% Q) * D) +
    as.vector(var_coef %*% diag(Q))
  
  trace_term <- numeric(nrow(D))
  linear_term <- numeric(nrow(D))
  for (j in seq_len(nrow(D))) {
    v <- pmax(var_coef[j, ], 0)
    trace_term[j] <- 2 * sum((Q^2) * tcrossprod(v, v))
    qd <- as.vector(Q %*% D[j, ])
    linear_term[j] <- 4 * sum(v * qd^2)
  }
  list(mu_J = mean_J, var_J = pmax(trace_term + linear_term, 1e-12))
}

objective_noise_variance <- function(
    mu_coef, sim, covariance = c("measurement", "total")) {
  covariance <- match.arg(covariance)
  Q <- sim$basis$loss_basis_cross
  Sigma <- if (covariance == "measurement") {
    sim$coefficient_measurement_noise_matrix
  } else {
    sim$coefficient_noise_matrix
  }
  QS <- Q %*% Sigma
  base <- 2 * sum(QS * t(QS))
  D <- sweep(mu_coef, 2, sim$target_coef, "-")
  qd <- D %*% Q
  pmax(base + 4 * rowSums((qd %*% Sigma) * qd), 1e-12)
}

sample_objective_draws <- function(pred, sim, n_draw, seed) {
  J <- nrow(pred$mu_coef)
  H <- ncol(pred$mu_coef)
  with_local_seed(seed, {
    Z <- matrix(rnorm(n_draw * H), nrow = n_draw, ncol = H)
    out <- matrix(0, nrow = J, ncol = n_draw)
    Q <- sim$basis$loss_basis_cross
    for (s in seq_len(n_draw)) {
      C <- pred$mu_coef +
        sweep(sqrt(pmax(pred$var_coef, 0)), 2, Z[s, ], "*")
      D <- sweep(C, 2, sim$target_coef, "-")
      out[, s] <- rowSums((D %*% Q) * D)
    }
    out
  })
}

make_simulation <- function(rep_id, cfg = CFG) {
  t <- seq(cfg$t_min, cfg$t_max, length.out = cfg$n_time)
  basis <- make_basis_objects(t, cfg)
  d <- length(cfg$input_lower)
  stopifnot(
    d == length(cfg$input_upper),
    d == length(cfg$target_u),
    all(cfg$input_upper > cfg$input_lower)
  )
  
  if (cfg$case_id == "section4") {
    stopifnot(d == 2L)
    axes <- Map(
      function(lo, hi) seq(lo, hi, length.out = cfg$grid_each),
      cfg$input_lower, cfg$input_upper
    )
    U <- as.matrix(expand.grid(axes))
    colnames(U) <- c("u0", "u1")
    U_unit <- sweep(
      sweep(U, 2, cfg$input_lower, "-"),
      2, cfg$input_upper - cfg$input_lower, "/"
    )
  } else if (cfg$case_id == "appendix6d") {
    stopifnot(d == 6L)
    U_unit <- lhs_unit(
      cfg$candidate_size - 1L, d,
      seed = cfg$seed + rep_id * 1009
    )
    target_unit <- (cfg$target_u - cfg$input_lower) /
      (cfg$input_upper - cfg$input_lower)
    U_unit <- rbind(target_unit, U_unit)
    U <- sweep(
      sweep(U_unit, 2, cfg$input_upper - cfg$input_lower, "*"),
      2, cfg$input_lower, "+"
    )
    colnames(U) <- paste0("u", seq_len(d))
  } else {
    stop("Unknown case_id: ", cfg$case_id)
  }
  colnames(U_unit) <- colnames(U)
  J <- nrow(U)
  
  X <- if (cfg$case_id == "section4") {
    matrix(exp(t), J, cfg$n_time, byrow = TRUE)
  } else {
    matrix(0, J, cfg$n_time)
  }
  
  x_center <- if (cfg$case_id == "section4") exp(t) else
    rep(0, cfg$n_time)
  x_scale <- if (cfg$case_id == "section4") {
    rep(cfg$functional_covariate_sd, cfg$n_time)
  } else {
    rep(1, cfg$n_time)
  }
  X_scaled <- sweep(sweep(X, 2, x_center, "-"), 2, x_scale, "/")
  
  true_mean <- matrix(NA_real_, J, cfg$n_time)
  for (j in seq_len(J)) {
    true_mean[j, ] <- true_mean_curve(U[j, ], t, cfg$scenario)
  }
  mean_input_signal_variance <- mean(apply(true_mean, 2, stats::var))
  
  target_curve <- true_mean_curve(cfg$target_u, t, cfg$scenario)
  target_coef <- as.vector(basis$A %*% target_curve)
  true_coef <- true_mean %*% t(basis$A)
  
  sim_for_loss <- list(
    basis = basis,
    target_coef = target_coef
  )
  true_J <- objective_from_coefficients(true_coef, sim_for_loss)
  true_best_idx <- which.min(true_J)
  
  x0 <- if (cfg$case_id == "section4") exp(t) else t
  K_process <- if (cfg$process_var > 0) {
    cfg$process_var *
      exp(-abs(outer(x0, x0, "-")) / cfg$process_range)
  } else {
    matrix(0, cfg$n_time, cfg$n_time)
  }
  Sigma_measurement <- diag(cfg$noise_sd^2, cfg$n_time)
  Sigma_obs <- K_process + Sigma_measurement
  coefficient_noise_matrix <- basis$A %*% Sigma_obs %*% t(basis$A)
  coefficient_measurement_noise_matrix <-
    basis$A %*% Sigma_measurement %*% t(basis$A)
  coefficient_noise <- diag(coefficient_noise_matrix)
  coefficient_noise <- pmax(coefficient_noise, 1e-10)
  
  if (cfg$diversity_reference >= J) {
    reference_idx <- seq_len(J)
  } else {
    ref_lhs <- lhs_unit(
      cfg$diversity_reference, d,
      seed = cfg$seed + rep_id * 2027
    )
    reference_idx <- match_lhs_to_candidates(ref_lhs, U_unit)
  }
  
  list(
    rep_id = rep_id, cfg = cfg, t = t, basis = basis,
    U = U, U_unit = U_unit, X = X, X_scaled = X_scaled,
    x_center = x_center, x_scale = x_scale,
    true_mean = true_mean, true_J = true_J,
    mean_input_signal_variance = mean_input_signal_variance,
    true_best_idx = true_best_idx,
    target_curve = target_curve, target_coef = target_coef,
    coefficient_noise = coefficient_noise,
    coefficient_noise_matrix = coefficient_noise_matrix,
    coefficient_measurement_noise_matrix =
      coefficient_measurement_noise_matrix,
    reference_idx = reference_idx
  )
}

simulate_observation <- function(sim, candidate_idx, eval_no) {
  cfg <- sim$cfg
  seed <- cfg$seed + sim$rep_id * 1000003 +
    candidate_idx * 1009 + eval_no * 97
  
  with_local_seed(seed, {
    x <- if (cfg$case_id == "section4") {
      exp(sim$t) + rnorm(1L, 0, cfg$functional_covariate_sd)
    } else {
      sim$X[candidate_idx, ]
    }
    x_scaled <- (x - sim$x_center) / sim$x_scale
    K_process <- if (cfg$process_var > 0) {
      cfg$process_var *
        exp(-abs(outer(x, x, "-")) / cfg$process_range)
    } else {
      matrix(0, cfg$n_time, cfg$n_time)
    }
    U <- safe_chol(
      K_process + diag(cfg$noise_sd^2, cfg$n_time),
      jitter = cfg$jitter
    )
    random_curve <- as.vector(t(U) %*% rnorm(cfg$n_time))
    list(
      y = sim$true_mean[candidate_idx, ] + random_curve,
      x = x,
      x_scaled = x_scaled
    )
  })
}

observe_indices <- function(sim, indices, initial_counts = NULL) {
  J <- nrow(sim$U)
  counts <- if (is.null(initial_counts)) integer(J) else initial_counts
  Y <- matrix(NA_real_, length(indices), sim$cfg$n_time)
  X_obs <- matrix(NA_real_, length(indices), sim$cfg$n_time)
  X_obs_scaled <- matrix(NA_real_, length(indices), sim$cfg$n_time)
  for (i in seq_along(indices)) {
    j <- indices[i]
    counts[j] <- counts[j] + 1L
    obs <- simulate_observation(sim, j, counts[j])
    Y[i, ] <- obs$y
    X_obs[i, ] <- obs$x
    X_obs_scaled[i, ] <- obs$x_scaled
  }
  list(
    Y = Y, X_obs = X_obs, X_obs_scaled = X_obs_scaled,
    counts = counts
  )
}

# ==============================================================================
# 3. Gaussian process model for functional responses
# ==============================================================================

trend_matrix <- function(U_unit, surrogate = "flexible") {
  U_unit <- as.matrix(U_unit)
  d <- ncol(U_unit)
  linear <- cbind(Intercept = 1, U_unit)
  colnames(linear) <- c("Intercept", paste0("u", seq_len(d)))
  if (surrogate == "linear") {
    return(linear)
  }
  if (surrogate != "flexible") stop("Unknown surrogate specification: ", surrogate)
  
  squares <- U_unit^2
  colnames(squares) <- paste0("u", seq_len(d), "_sq")
  interactions <- NULL
  if (d >= 2L) {
    pairs <- combn(d, 2)
    interactions <- vapply(seq_len(ncol(pairs)), function(k) {
      U_unit[, pairs[1, k]] * U_unit[, pairs[2, k]]
    }, numeric(nrow(U_unit)))
    if (is.vector(interactions)) {
      interactions <- matrix(interactions, ncol = 1L)
    }
    colnames(interactions) <- apply(pairs, 2, function(z) {
      paste0("u", z[1], "_u", z[2])
    })
  }
  cbind(linear, squares, interactions)
}

kernel_from_features <- function(Ua, Xa, Ub, Xb, hyper, surrogate) {
  Ua <- as.matrix(Ua)
  Ub <- as.matrix(Ub)
  Xa <- as.matrix(Xa)
  Xb <- as.matrix(Xb)
  sqdist <- function(A, B) {
    pmax(outer(rowSums(A^2), rowSums(B^2), "+") -
           2 * tcrossprod(A, B), 0)
  }
  
  Ua_scaled <- sweep(Ua, 2, hyper$ell_u, "/")
  Ub_scaled <- sweep(Ub, 2, hyper$ell_u, "/")
  d_u <- if (surrogate == "linear") {
    matrix(0, nrow(Ua), nrow(Ub))
  } else {
    sqdist(Ua_scaled, Ub_scaled)
  }
  
  d_x <- sqdist(Xa, Xb) / ncol(Xa) / hyper$ell_x^2
  exp(-0.5 * (d_u + d_x))
}

kernel_matrix <- function(idx_a, idx_b, sim, hyper) {
  kernel_from_features(
    sim$U_unit[idx_a, , drop = FALSE],
    sim$X_scaled[idx_a, , drop = FALSE],
    sim$U_unit[idx_b, , drop = FALSE],
    sim$X_scaled[idx_b, , drop = FALSE],
    hyper, sim$cfg$surrogate
  )
}

hyper_score <- function(C, obs_idx, X_obs_scaled, sim, hyper) {
  R <- kernel_from_features(
    sim$U_unit[obs_idx, , drop = FALSE], X_obs_scaled,
    sim$U_unit[obs_idx, , drop = FALSE], X_obs_scaled,
    hyper, sim$cfg$surrogate
  )
  Hm <- trend_matrix(
    sim$U_unit[obs_idx, , drop = FALSE], sim$cfg$surrogate
  )
  score <- 0
  
  for (h in seq_len(ncol(C))) {
    y <- C[, h]
    beta_ols <- solve(crossprod(Hm) + diag(1e-8, ncol(Hm)),
                      crossprod(Hm, y))
    residual <- y - as.vector(Hm %*% beta_ols)
    noise <- sim$coefficient_noise[h]
    signal <- max(var(residual) - noise, 0.05 * noise, 1e-8)
    K <- signal * R + diag(noise, length(y))
    U <- safe_chol(K, jitter = sim$cfg$jitter)
    Kinv_y <- chol_solve(U, residual)
    score <- score + 0.5 * sum(residual * Kinv_y) +
      sum(log(diag(U)))
  }
  score
}

tune_kernel <- function(Y, obs_idx, X_obs_scaled, sim) {
  cfg <- sim$cfg
  if (!cfg$tune_hyper) {
    return(list(
      ell_u = rep(cfg$kernel_ell_u, ncol(sim$U_unit)),
      ell_x = cfg$kernel_ell_x
    ))
  }
  
  C <- Y %*% t(sim$basis$A)
  candidates <- expand.grid(
    ell_u = cfg$hyper_grid_u,
    ell_x = cfg$hyper_grid_x
  )
  scores <- numeric(nrow(candidates))
  for (i in seq_len(nrow(candidates))) {
    hyper <- list(
      ell_u = rep(candidates$ell_u[i], ncol(sim$U_unit)),
      ell_x = candidates$ell_x[i]
    )
    scores[i] <- hyper_score(C, obs_idx, X_obs_scaled, sim, hyper)
  }
  best <- which.min(scores)
  list(
    ell_u = rep(candidates$ell_u[best], ncol(sim$U_unit)),
    ell_x = candidates$ell_x[best]
  )
}

fit_functional_gp <- function(Y, obs_idx, X_obs_scaled, sim, hyper) {
  C <- Y %*% t(sim$basis$A)
  Hm <- trend_matrix(
    sim$U_unit[obs_idx, , drop = FALSE], sim$cfg$surrogate
  )
  R <- kernel_from_features(
    sim$U_unit[obs_idx, , drop = FALSE], X_obs_scaled,
    sim$U_unit[obs_idx, , drop = FALSE], X_obs_scaled,
    hyper, sim$cfg$surrogate
  )
  fits <- vector("list", ncol(C))
  
  for (h in seq_len(ncol(C))) {
    y <- C[, h]
    noise <- sim$coefficient_noise[h]
    
    beta_ols <- solve(crossprod(Hm) + diag(1e-8, ncol(Hm)),
                      crossprod(Hm, y))
    residual_ols <- y - as.vector(Hm %*% beta_ols)
    signal <- max(var(residual_ols) - noise, 0.05 * noise, 1e-8)
    
    K <- signal * R + diag(noise, length(y))
    Uchol <- safe_chol(K, jitter = sim$cfg$jitter)
    Kinv <- chol2inv(Uchol)
    XtKiX <- crossprod(Hm, Kinv %*% Hm)
    Cbeta <- solve(XtKiX + diag(1e-10, ncol(Hm)))
    beta <- Cbeta %*% crossprod(Hm, Kinv %*% y)
    residual <- y - as.vector(Hm %*% beta)
    alpha <- Kinv %*% residual
    
    fits[[h]] <- list(
      signal = signal, noise = noise, beta = beta, alpha = alpha,
      Kinv = Kinv, H_train = Hm, Cbeta = Cbeta
    )
  }
  
  list(
    fits = fits, obs_idx = obs_idx, hyper = hyper,
    X_obs_scaled = X_obs_scaled,
    sim = sim, coefficient_observed = C
  )
}

predict_functional_gp <- function(model, pred_idx, return_curve = FALSE) {
  sim <- model$sim
  Hs <- trend_matrix(
    sim$U_unit[pred_idx, , drop = FALSE], sim$cfg$surrogate
  )
  Rst <- kernel_from_features(
    sim$U_unit[pred_idx, , drop = FALSE],
    sim$X_scaled[pred_idx, , drop = FALSE],
    sim$U_unit[model$obs_idx, , drop = FALSE],
    model$X_obs_scaled,
    model$hyper, sim$cfg$surrogate
  )
  Hcoef <- length(model$fits)
  mu_coef <- matrix(NA_real_, length(pred_idx), Hcoef)
  var_coef <- matrix(NA_real_, length(pred_idx), Hcoef)
  
  for (h in seq_len(Hcoef)) {
    fit <- model$fits[[h]]
    Kst <- fit$signal * Rst
    mu_coef[, h] <- as.vector(Hs %*% fit$beta + Kst %*% fit$alpha)
    
    base_reduction <- rowSums((Kst %*% fit$Kinv) * Kst)
    r <- Hs - Kst %*% fit$Kinv %*% fit$H_train
    trend_var <- rowSums((r %*% fit$Cbeta) * r)
    var_coef[, h] <- pmax(
      fit$signal - base_reduction + trend_var, 1e-12
    )
  }
  
  obj <- objective_moments(mu_coef, var_coef, sim)
  
  ans <- list(
    mu_coef = mu_coef, var_coef = var_coef,
    mu_J = obj$mu_J, var_J = obj$var_J
  )
  
  if (return_curve) {
    B <- sim$basis$B
    ans$mu_curve <- mu_coef %*% t(B)
    ans$var_curve <- var_coef %*% t(B^2)
    ans$sd_curve <- sqrt(pmax(ans$var_curve, 0))
  }
  ans
}

posterior_crosscov <- function(model, coefficient, idx_a, idx_b) {
  sim <- model$sim
  fit <- model$fits[[coefficient]]
  R_ab <- kernel_matrix(idx_a, idx_b, sim, model$hyper)
  R_at <- kernel_from_features(
    sim$U_unit[idx_a, , drop = FALSE],
    sim$X_scaled[idx_a, , drop = FALSE],
    sim$U_unit[model$obs_idx, , drop = FALSE],
    model$X_obs_scaled,
    model$hyper, sim$cfg$surrogate
  )
  R_bt <- kernel_from_features(
    sim$U_unit[idx_b, , drop = FALSE],
    sim$X_scaled[idx_b, , drop = FALSE],
    sim$U_unit[model$obs_idx, , drop = FALSE],
    model$X_obs_scaled,
    model$hyper, sim$cfg$surrogate
  )
  K_at <- fit$signal * R_at
  K_bt <- fit$signal * R_bt
  
  Ha <- trend_matrix(
    sim$U_unit[idx_a, , drop = FALSE], sim$cfg$surrogate
  )
  Hb <- trend_matrix(
    sim$U_unit[idx_b, , drop = FALSE], sim$cfg$surrogate
  )
  ra <- Ha - K_at %*% fit$Kinv %*% fit$H_train
  rb <- Hb - K_bt %*% fit$Kinv %*% fit$H_train
  
  fit$signal * R_ab -
    K_at %*% fit$Kinv %*% t(K_bt) +
    ra %*% fit$Cbeta %*% t(rb)
}

# ==============================================================================
# 4. Acquisition functions
# ==============================================================================

# Attenuation is activated only when posterior objective uncertainty is no
# greater than the propagated measurement-noise variance.
weight_decay <- function(pred_all, obs_idx, counts, sim) {
  cfg <- sim$cfg
  unique_idx <- which(counts > 0L)
  objective_noise <- objective_noise_variance(pred_all$mu_coef, sim)
  tau_J <- cfg$kappa * objective_noise
  active <- pred_all$var_J[unique_idx] <= tau_J[unique_idx]
  
  if (!any(active)) {
    return(list(W = rep(1, nrow(sim$U)), active = active, tau_J = tau_J))
  }
  
  log_W <- rep(0, nrow(sim$U))
  for (k in which(active)) {
    idx <- unique_idx[k]
    d2 <- rowSums((sweep(sim$U_unit, 2, sim$U_unit[idx, ], "-"))^2)
    local_weight <- 1 - (1 - cfg$rho) * exp(-cfg$eta * d2)
    local_weight <- pmax(local_weight, 1e-12)
    
    log_W <- log_W + counts[idx] * log(local_weight)
  }
  list(W = exp(log_W), active = active, tau_J = tau_J)
}

global_profile_learning_gain <- function(model, pred_all, sim) {
  ref <- sim$reference_idx
  all_idx <- seq_len(nrow(sim$U))
  H <- length(model$fits)
  J <- length(all_idx)
  
  cross_array <- array(NA_real_, dim = c(length(ref), J, H))
  for (h in seq_along(model$fits)) {
    cross_array[, , h] <- posterior_crosscov(
      model, h, ref, all_idx
    )
  }
  
  Q_B <- sim$basis$integrated_basis_cross
  D <- numeric(J)
  for (j in seq_len(J)) {
    K_future <- diag(pred_all$var_coef[j, ], H) +
      sim$coefficient_noise_matrix
    U_future <- safe_chol(K_future, jitter = sim$cfg$jitter)
    K_future_inv <- chol2inv(U_future)
    cross_j <- matrix(
      cross_array[, j, ], nrow = length(ref), ncol = H
    )
    cross_gram <- crossprod(cross_j) / length(ref)
    D[j] <- sum(Q_B * K_future_inv * cross_gram)
  }
  pmax(D, 0)
}

expected_quantile_improvement <- function(pred_all, obs_unique_idx,
                                          objective_noise, beta = 0.90) {
  z_beta <- qnorm(beta)
  q_current <- pred_all$mu_J +
    z_beta * sqrt(pmax(pred_all$var_J, 0))
  q_incumbent <- min(q_current[obs_unique_idx])
  
  v <- pmax(pred_all$var_J, 1e-12)
  noise <- pmax(objective_noise, 1e-12)
  v_after <- pmax(v - v^2 / (v + noise), 0)
  sd_mean_update <- v / sqrt(v + noise)
  target_after_mean <- pred_all$mu_J + z_beta * sqrt(v_after)
  delta <- q_incumbent - target_after_mean
  
  ans <- numeric(length(delta))
  regular <- sd_mean_update > 1e-12
  z <- delta[regular] / sd_mean_update[regular]
  ans[regular] <- delta[regular] * pnorm(z) +
    sd_mean_update[regular] * dnorm(z)
  ans[!regular] <- pmax(delta[!regular], 0)
  pmax(ans, 0)
}

functional_acquisition <- function(method, model, counts, sim,
                                   iteration = 0L) {
  all_idx <- seq_len(nrow(sim$U))
  pred <- predict_functional_gp(model, all_idx)
  observed_unique <- which(counts > 0L)
  
  incumbent <- min(pred$mu_J[observed_unique])
  objective_draws <- sample_objective_draws(
    pred, sim, sim$cfg$objective_mc,
    seed = sim$cfg$seed + sim$rep_id * 104729 +
      length(model$obs_idx) * 1009 + iteration * 97
  )
  FEI <- rowMeans(pmax(incumbent - objective_draws, 0))
  
  decay <- weight_decay(pred, model$obs_idx, counts, sim)
  W <- decay$W
  D <- if (method %in% c("FEI+D", "WDEI")) {
    global_profile_learning_gain(model, pred, sim)
  } else {
    rep(0, length(FEI))
  }
  
  if (method == "GPFR-FEI") {
    score <- FEI
  } else if (method == "FEI+W") {
    score <- W * FEI
  } else if (method == "FEI+D") {
    c_I <- max(FEI) + sim$cfg$utility_epsilon
    c_D <- max(D) + sim$cfg$utility_epsilon
    score <- (1 - sim$cfg$alpha) * FEI / c_I +
      sim$cfg$alpha * D / c_D
  } else if (method == "WDEI") {
    FEI_w <- W * FEI
    c_I <- max(FEI_w) + sim$cfg$utility_epsilon
    c_D <- max(D) + sim$cfg$utility_epsilon
    score <- (1 - sim$cfg$alpha) * FEI_w / c_I +
      sim$cfg$alpha * D / c_D
  } else if (method == "GPFR-EQI") {
    objective_noise <- objective_noise_variance(
      pred$mu_coef, sim, covariance = "total"
    )
    score <- expected_quantile_improvement(
      pred, observed_unique, objective_noise,
      beta = sim$cfg$eqi_beta
    )
  } else {
    stop("Unknown acquisition method: ", method)
  }
  
  list(
    score = score, FEI = FEI, W = W, D = D,
    pred = pred, incumbent = incumbent,
    tau_J = decay$tau_J,
    active_decay = sum(decay$active),
    integrated_variance = mean(
      as.vector(pred$var_coef %*% sim$basis$integrated_basis_sq)
    )
  )
}

# ==============================================================================
# 5. Sequential evaluation and performance metrics
# ==============================================================================

select_recommendation <- function(model, candidate_idx, sim,
                                  return_curve = FALSE) {
  pred <- predict_functional_gp(
    model, candidate_idx, return_curve = return_curve
  )
  score <- pred$mu_J
  local <- which.min(score)
  list(
    local = local, index = candidate_idx[local],
    score = score, pred = pred
  )
}

run_sequential_method <- function(method, sim, initial_idx, initial_Y,
                                  initial_X_obs_scaled, hyper) {
  cfg <- sim$cfg
  obs_idx <- initial_idx
  Y <- initial_Y
  X_obs_scaled <- initial_X_obs_scaled
  counts <- tabulate(obs_idx, nbins = nrow(sim$U))
  n_add <- cfg$N_total - length(initial_idx)
  
  history <- data.frame(
    Iteration = seq_len(n_add),
    Selected = NA_integer_,
    MaxAcquisition = NA_real_,
    MaxFEI = NA_real_,
    MaxWeightedFEI = NA_real_,
    MaxD = NA_real_,
    SelectedW = NA_real_,
    SelectedD = NA_real_,
    ActiveDecay = NA_integer_,
    IntegratedVariance = NA_real_,
    SelectedVarJ = NA_real_,
    TauJ = NA_real_,
    IsRepeat = NA_integer_,
    UsefulRepeat = NA_integer_,
    RedundantRepeat = NA_integer_,
    CumulativeReplicates = NA_integer_,
    CumulativeRedundantRepeats = NA_integer_,
    OracleBestRegret = NA_real_,
    RecommendationRegret = NA_real_,
    Seconds = NA_real_
  )
  
  model <- fit_functional_gp(Y, obs_idx, X_obs_scaled, sim, hyper)
  
  for (iteration in seq_len(n_add)) {
    start <- proc.time()[3]
    
    acq <- functional_acquisition(
      method, model, counts, sim, iteration = iteration
    )
    score <- acq$score
    FEI <- acq$FEI
    W <- acq$W
    D <- acq$D
    active_decay <- acq$active_decay
    integrated_variance <- acq$integrated_variance
    pred_current <- acq$pred
    tau_current <- acq$tau_J
    
    if (!any(is.finite(score))) {
      stop("All candidate acquisition values are invalid; method: ", method,
           ", replication: ", sim$rep_id, ", iteration: ", iteration)
    }
    selected <- which.max(score)
    is_repeat <- as.integer(counts[selected] > 0L)
    selected_var_j <- pred_current$var_J[selected]
    useful_repeat <- as.integer(
      is_repeat == 1L && selected_var_j > tau_current[selected]
    )
    redundant_repeat <- as.integer(
      is_repeat == 1L && selected_var_j <= tau_current[selected]
    )
    
    counts[selected] <- counts[selected] + 1L
    obs_new <- simulate_observation(sim, selected, counts[selected])
    obs_idx <- c(obs_idx, selected)
    Y <- rbind(Y, obs_new$y)
    X_obs_scaled <- rbind(X_obs_scaled, obs_new$x_scaled)
    
    model <- fit_functional_gp(Y, obs_idx, X_obs_scaled, sim, hyper)
    evaluated_unique <- which(counts > 0L)
    rec <- select_recommendation(model, evaluated_unique, sim)
    rec_idx <- rec$index
    
    history$Selected[iteration] <- selected
    history$MaxAcquisition[iteration] <- max(score, na.rm = TRUE)
    history$MaxFEI[iteration] <- if (all(is.na(FEI))) NA_real_ else
      max(FEI, na.rm = TRUE)
    history$MaxWeightedFEI[iteration] <- if (
      all(is.na(FEI)) || all(is.na(W))
    ) NA_real_ else max(W * FEI, na.rm = TRUE)
    history$MaxD[iteration] <- if (all(is.na(D))) NA_real_ else
      max(D, na.rm = TRUE)
    history$SelectedW[iteration] <- W[selected]
    history$SelectedD[iteration] <- D[selected]
    history$ActiveDecay[iteration] <- active_decay
    history$IntegratedVariance[iteration] <- integrated_variance
    history$SelectedVarJ[iteration] <- selected_var_j
    history$TauJ[iteration] <- tau_current[selected]
    history$IsRepeat[iteration] <- is_repeat
    history$UsefulRepeat[iteration] <- useful_repeat
    history$RedundantRepeat[iteration] <- redundant_repeat
    history$CumulativeReplicates[iteration] <-
      sum(pmax(counts - 1L, 0L))
    history$CumulativeRedundantRepeats[iteration] <-
      sum(history$RedundantRepeat[seq_len(iteration)], na.rm = TRUE)
    history$OracleBestRegret[iteration] <-
      min(sim$true_J[which(counts > 0L)]) -
      sim$true_J[sim$true_best_idx]
    history$RecommendationRegret[iteration] <-
      sim$true_J[rec_idx] - sim$true_J[sim$true_best_idx]
    history$Seconds[iteration] <- proc.time()[3] - start
    
    if (cfg$verbose && iteration %% 5L == 0L) {
      cat(sprintf("  %-10s: completed %2d/%2d sequential evaluations\n",
                  method, iteration, n_add))
    }
  }
  
  list(
    method = method, obs_idx = obs_idx, Y = Y,
    X_obs_scaled = X_obs_scaled,
    counts = counts, model = model, history = history
  )
}

evaluate_method <- function(method, model, obs_idx, counts, sim, history = NULL) {
  all_idx <- seq_len(nrow(sim$U))
  pred_all <- predict_functional_gp(model, all_idx, return_curve = TRUE)
  
  curve_error <- pred_all$mu_curve - sim$true_mean
  curve_rmse <- sqrt(rowMeans(curve_error^2))
  global_rmse <- mean(curve_rmse)
  integrated_var <- mean(
    pred_all$var_curve * matrix(
      sim$basis$domain_weights,
      nrow = nrow(sim$U), ncol = sim$cfg$n_time, byrow = TRUE
    )
  ) * sim$cfg$n_time
  
  evaluated_unique <- which(counts > 0L)
  rec <- select_recommendation(
    model, evaluated_unique, sim, return_curve = TRUE
  )
  pred_evaluated <- rec$pred
  rec_local <- rec$local
  rec_idx <- rec$index
  
  true_regret <- sim$true_J[rec_idx] - sim$true_J[sim$true_best_idx]
  oracle_pog <- min(sim$true_J[evaluated_unique]) -
    sim$true_J[sim$true_best_idx]
  pog <- true_regret
  opt_rmse <- sqrt(mean(
    (pred_evaluated$mu_curve[rec_local, ] -
       sim$true_mean[rec_idx, ])^2
  ))
  curve_gap <- sqrt(mean(
    (sim$true_mean[rec_idx, ] -
       sim$true_mean[sim$true_best_idx, ])^2
  ))
  input_error <- sqrt(sum(
    (sim$U[rec_idx, ] -
       sim$U[sim$true_best_idx, ])^2
  ))
  normalized_input_error <- sqrt(sum(
    (sim$U_unit[rec_idx, ] -
       sim$U_unit[sim$true_best_idx, ])^2
  ))
  data.frame(
    Rep = sim$rep_id,
    Method = method,
    Global_RMSE = global_rmse,
    Integrated_Variance = integrated_var,
    POG = pog,
    Oracle_POG = oracle_pog,
    Opt_RMSE = opt_rmse,
    True_Curve_Gap = curve_gap,
    Input_Error = input_error,
    Normalized_Input_Error = normalized_input_error,
    Unique_Inputs = length(evaluated_unique),
    Replicates = length(obs_idx) - length(evaluated_unique),
    Useful_Repeats = if (is.null(history)) NA_real_ else
      sum(history$UsefulRepeat, na.rm = TRUE),
    Redundant_Repeats = if (is.null(history)) NA_real_ else
      sum(history$RedundantRepeat, na.rm = TRUE),
    Activation_Rate = if (
      is.null(history) || all(is.na(history$ActiveDecay))
    ) NA_real_ else mean(history$ActiveDecay > 0, na.rm = TRUE),
    Effective_Repeat_Rate = if (
      is.null(history) || sum(history$IsRepeat) == 0L
    ) NA_real_ else
      sum(history$UsefulRepeat) / sum(history$IsRepeat),
    Recommendation_Posterior_Mean = rec$score[rec_local],
    Mean_Iteration_Seconds = if (is.null(history)) NA_real_ else
      mean(history$Seconds),
    Recommended_Index = rec_idx,
    Recommended_u0 = sim$U[rec_idx, 1],
    Recommended_u1 = sim$U[rec_idx, 2],
    True_Best_u0 = sim$U[sim$true_best_idx, 1],
    True_Best_u1 = sim$U[sim$true_best_idx, 2],
    Recommended_Input = paste(sprintf("%.5f", sim$U[rec_idx, ]),
                              collapse = ","),
    True_Best_Input = paste(
      sprintf("%.5f", sim$U[sim$true_best_idx, ]), collapse = ","
    ),
    Process_Variance = sim$cfg$process_var,
    Measurement_Noise_SD = sim$cfg$noise_sd,
    Kappa = sim$cfg$kappa,
    Rho = sim$cfg$rho,
    Eta = sim$cfg$eta,
    Eta0 = sim$eta_scale$eta0,
    Initial_NN_Radius = sim$eta_scale$r0,
    Scenario = sim$cfg$scenario,
    Surrogate = sim$cfg$surrogate
  )
}

run_replication <- function(rep_id, methods, cfg = CFG,
                            keep_details = FALSE) {
  sim <- make_simulation(rep_id, cfg)
  
  initial_lhs <- lhs_unit(
    cfg$M_init, ncol(sim$U_unit),
    seed = cfg$seed + rep_id * 7919
  )
  initial_idx <- match_lhs_to_candidates(initial_lhs, sim$U_unit)
  
  eta_scale <- derive_eta_from_initial_design(
    sim$U_unit, initial_idx, cfg
  )
  sim$cfg$eta <- eta_scale$eta
  sim$eta_scale <- eta_scale
  
  initial_obs <- observe_indices(sim, initial_idx)
  initial_Y <- initial_obs$Y
  initial_X_obs_scaled <- initial_obs$X_obs_scaled
  
  hyper <- tune_kernel(
    initial_Y, initial_idx, initial_X_obs_scaled, sim
  )
  
  metrics <- list()
  details <- list()
  histories <- list()
  counter <- 1L
  history_counter <- 1L
  
  for (method in methods) {
    if (cfg$verbose) {
      cat(sprintf("\nReplication %d, method: %s\n", rep_id, method))
    }
    
    if (method == "Initial LHS") {
      counts <- tabulate(initial_idx, nbins = nrow(sim$U))
      model <- fit_functional_gp(
        initial_Y, initial_idx, initial_X_obs_scaled, sim, hyper
      )
      metrics[[counter]] <- evaluate_method(
        method, model, initial_idx, counts, sim
      )
      if (keep_details) {
        details[[method]] <- list(
          method = method, obs_idx = initial_idx, Y = initial_Y,
          X_obs_scaled = initial_X_obs_scaled,
          counts = counts, model = model, history = NULL
        )
      }
      
    } else if (method == "One-shot LHS") {
      full_idx <- maximin_lhs_indices(
        cfg$N_total, sim$U_unit,
        n_starts = 200L,
        seed = cfg$seed + rep_id * 3571
      )
      full_obs <- observe_indices(sim, full_idx)
      full_Y <- full_obs$Y
      counts <- full_obs$counts
      tune_n <- min(cfg$M_init, cfg$N_total)
      hyper_os <- tune_kernel(
        full_Y[seq_len(tune_n), , drop = FALSE],
        full_idx[seq_len(tune_n)],
        full_obs$X_obs_scaled[seq_len(tune_n), , drop = FALSE],
        sim
      )
      model <- fit_functional_gp(
        full_Y, full_idx, full_obs$X_obs_scaled, sim, hyper_os
      )
      metrics[[counter]] <- evaluate_method(
        method, model, full_idx, counts, sim
      )
      if (keep_details) {
        details[[method]] <- list(
          method = method, obs_idx = full_idx, Y = full_Y,
          X_obs_scaled = full_obs$X_obs_scaled,
          counts = counts, model = model, history = NULL
        )
      }
      
    } else {
      fit <- run_sequential_method(
        method, sim, initial_idx, initial_Y,
        initial_X_obs_scaled, hyper
      )
      metrics[[counter]] <- evaluate_method(
        method, fit$model, fit$obs_idx, fit$counts,
        sim, fit$history
      )
      history_block <- fit$history
      history_block$Rep <- rep_id
      history_block$Method <- method
      histories[[history_counter]] <- history_block
      history_counter <- history_counter + 1L
      if (keep_details) details[[method]] <- fit
    }
    counter <- counter + 1L
  }
  
  list(
    metrics = do.call(rbind, metrics),
    history = if (length(histories) == 0L) NULL else
      do.call(rbind, histories),
    details = details,
    sim = if (keep_details) sim else NULL,
    hyper = hyper
  )
}

run_experiment <- function(n_rep, methods, cfg = CFG,
                           keep_first = TRUE, label = "",
                           representative_method = "WDEI") {
  stopifnot(n_rep == 1L)
  all_metrics <- vector("list", n_rep)
  all_histories <- vector("list", n_rep)
  first_details <- NULL
  start <- proc.time()[3]
  
  for (r in seq_len(n_rep)) {
    cat(sprintf("\n%s: independent replication %d/%d\n", label, r, n_rep))
    ans <- run_replication(
      rep_id = r, methods = methods, cfg = cfg,
      keep_details = keep_first && r == 1L
    )
    all_metrics[[r]] <- ans$metrics
    all_histories[[r]] <- ans$history
    if (keep_first && r == 1L) first_details <- ans
  }
  
  raw <- do.call(rbind, all_metrics)
  list(
    raw = raw,
    history = if (all(vapply(all_histories, is.null, logical(1)))) NULL else
      do.call(rbind, all_histories[!vapply(
        all_histories, is.null, logical(1)
      )]),
    first = first_details,
    representative_rep = 1L,
    seconds = proc.time()[3] - start
  )
}

# ==============================================================================
# 6. Tables and figures
# ==============================================================================

print_main_tables <- function(raw) {
  methods <- unique(raw$Method)
  main_table <- do.call(rbind, lapply(methods, function(method) {
    z <- raw[raw$Method == method, , drop = FALSE]
    cell <- function(x, digits = 4) {
      sprintf(paste0("%.", digits, "f"), x[1])
    }
    data.frame(
      Method = method,
      Budget = if (method == "Initial LHS") min(z$Unique_Inputs) else
        max(z$Unique_Inputs + z$Replicates),
      Global_RMSE = cell(z$Global_RMSE),
      Local_RMSE = cell(z$Opt_RMSE),
      POG = cell(z$POG),
      Replicates = cell(z$Replicates, 2),
      check.names = FALSE
    )
  }))
  
  cat("\n\n================ Section 4 results: single-run values ================\n")
  print(main_table, row.names = FALSE, right = FALSE)
  cat("\nNote: POG is the functional-loss difference between the recommended attainable input and the true optimal attainable input.\n")
  
  diagnostics <- summarize_metrics(
    raw,
    c("Integrated_Variance", "Unique_Inputs", "Effective_Repeat_Rate",
      "Mean_Iteration_Seconds")
  )
  cat("\n================ Supplementary diagnostics: single-run values ================\n")
  print(diagnostics, row.names = FALSE)
  invisible(list(main = main_table, diagnostics = diagnostics))
}

plot_main_boxplots <- function(raw, history = NULL) {
  oldpar <- par(no.readonly = TRUE)
  on.exit(par(oldpar), add = TRUE)
  set_plot_style(mfrow = c(2, 2), mar = c(5.8, 4.8, 1.5, 0.6))
  method_order <- c("Initial LHS", "One-shot LHS", "GPFR-EQI", "WDEI")
  raw <- raw[match(method_order, raw$Method), , drop = FALSE]
  short_names <- c("Initial\nLHS", "One-shot\nLHS", "GPFR-EQI", "WDEI")
  
  draw_points <- function(values, ylab, col) {
    plot(seq_along(values), values, xaxt = "n", type = "b",
         pch = 19, lwd = 1.6, col = col, ylab = ylab, xlab = "")
    axis(1, at = seq_along(short_names), labels = short_names,
         tick = FALSE, cex.axis = 0.88, line = -0.5)
    grid(nx = NA, ny = NULL, col = "grey88")
  }
  draw_points(raw$Global_RMSE, "Global RMSE", "#1F78B4")
  draw_points(raw$Opt_RMSE, "Local RMSE", "#FF7F00")
  draw_points(raw$POG, "Feasible optimality gap", "#33A02C")
  
  if (!is.null(history)) {
    plot_convergence_history(history, add_panel = TRUE)
  } else {
    plot.new()
    text(0.5, 0.5, "No sequential history")
  }
  invisible(NULL)
}

summarize_history <- function(history, metric = "RecommendationRegret") {
  stopifnot(!is.null(history), metric %in% names(history))
  keys <- unique(history[, c("Method", "Iteration")])
  rows <- vector("list", nrow(keys))
  for (i in seq_len(nrow(keys))) {
    z <- history[
      history$Method == keys$Method[i] &
        history$Iteration == keys$Iteration[i], metric
    ]
    rows[[i]] <- data.frame(
      Method = keys$Method[i],
      Iteration = keys$Iteration[i],
      Value = z[1],
      row.names = NULL
    )
  }
  do.call(rbind, rows)
}

plot_feasible_objective <- function(first_result) {
  if (is.null(first_result)) return(invisible(NULL))
  sim <- first_result$sim
  oldpar <- par(no.readonly = TRUE)
  on.exit(par(oldpar), add = TRUE)
  set_plot_style(mfrow = c(1, 2), mar = c(4.2, 4.8, 2.0, 0.8))
  
  nearest_idx <- function(target) {
    which.min(rowSums((sweep(sim$U, 2, target, "-"))^2))
  }
  show_idx <- unique(c(
    nearest_idx(c(-1, 6)), nearest_idx(c(-1, 8)),
    nearest_idx(c(1, 6)), nearest_idx(c(1, 8)),
    sim$true_best_idx
  ))
  cols <- c("#D95F75", "#A58A00", "#00A65A", "#0099CC", "#AA66CC")
  yrange <- range(sim$true_mean[show_idx, , drop = FALSE])
  plot(
    sim$t, sim$true_mean[show_idx[1], ], type = "l",
    lwd = 2, col = cols[1], ylim = yrange,
    xlab = "Functional index t", ylab = "Latent mean response"
  )
  if (length(show_idx) > 1L) {
    for (i in 2:length(show_idx)) {
      lines(sim$t, sim$true_mean[show_idx[i], ],
            lwd = 2, col = cols[i])
    }
  }
  labels <- sprintf(
    "(%.2f, %.2f)%s",
    sim$U[show_idx, 1], sim$U[show_idx, 2],
    ifelse(show_idx == sim$true_best_idx, "  optimum", "")
  )
  legend("topleft", legend = labels, col = cols, lwd = 2,
         bty = "n", cex = 0.80)
  grid(col = "grey88")
  
  u0 <- sort(unique(sim$U[, 1]))
  u1 <- sort(unique(sim$U[, 2]))
  z <- matrix(sim$true_J, nrow = length(u0), ncol = length(u1))
  contour(
    u0, u1, z, nlevels = 14, drawlabels = TRUE,
    xlab = expression(u[0]), ylab = expression(u[1]),
    col = "#4D4D4D"
  )
  points(
    sim$U[sim$true_best_idx, 1],
    sim$U[sim$true_best_idx, 2],
    pch = 8, cex = 1.5, lwd = 2, col = "#D73027"
  )
  title(sprintf(
    "Feasible optimum: (%.2f, %.2f)",
    sim$U[sim$true_best_idx, 1],
    sim$U[sim$true_best_idx, 2]
  ))
  invisible(NULL)
}

plot_convergence_history <- function(history, add_panel = FALSE) {
  if (is.null(history)) return(invisible(NULL))
  summary <- summarize_history(history, "RecommendationRegret")
  keep <- summary$Method %in% c("GPFR-EQI", "WDEI")
  summary <- summary[keep, , drop = FALSE]
  methods <- unique(summary$Method)
  cols <- c("GPFR-EQI" = "#33A02C", "WDEI" = "#1696D2")[methods]
  if (!add_panel) {
    oldpar <- par(no.readonly = TRUE)
    on.exit(par(oldpar), add = TRUE)
    set_plot_style()
  }
  ylim <- range(pmax(summary$Value, 0), finite = TRUE)
  first <- summary[summary$Method == methods[1], ]
  plot(
    first$Iteration, first$Value, type = "n", ylim = ylim,
    xlab = "Sequential iteration", ylab = "Recommendation POG"
  )
  for (i in seq_along(methods)) {
    sub <- summary[summary$Method == methods[i], ]
    sub <- sub[order(sub$Iteration), ]
    lines(sub$Iteration, sub$Value, lwd = 2, col = cols[i])
  }
  legend("topright", legend = methods, col = cols,
         lwd = 2, bty = "n", cex = 0.90)
  grid(col = "grey88")
  invisible(summary)
}

plot_first_replication <- function(first_result) {
  if (is.null(first_result)) return(invisible(NULL))
  sim <- first_result$sim
  details <- first_result$details
  oldpar <- par(no.readonly = TRUE)
  on.exit(par(oldpar), add = TRUE)
  
  if ("WDEI" %in% names(details)) {
    h <- details[["WDEI"]]$history
    set_plot_style(mfrow = c(2, 2), mar = c(4.0, 4.7, 1.5, 0.6))
    plot(h$Iteration, h$MaxFEI, type = "o", pch = 19,
         col = "#1F78B4", xlab = "Iteration", ylab = "Maximum FEI")
    grid(col = "grey85")
    plot(h$Iteration, h$IntegratedVariance, type = "o", pch = 19,
         col = "#33A02C", xlab = "Iteration",
         ylab = "Integrated latent variance")
    grid(col = "grey85")
    plot(h$Iteration, h$ActiveDecay, type = "o", pch = 19,
         col = "#E31A1C", xlab = "Iteration",
         ylab = "Activated decay locations")
    grid(col = "grey85")
    plot(h$Iteration, h$CumulativeReplicates, type = "o", pch = 19,
         col = "#6A3D9A", xlab = "Iteration",
         ylab = "Cumulative replications")
    grid(col = "grey85")
  }
  
  method_names <- names(details)
  n_method <- length(method_names)
  nr <- ceiling(sqrt(n_method))
  nc <- ceiling(n_method / nr)
  set_plot_style(
    mfrow = c(nr, nc), mar = c(3.8, 4.2, 1.8, 0.5),
    oma = c(0.5, 0.5, 0, 0)
  )
  for (method in method_names) {
    fit <- details[[method]]
    evaluated <- which(fit$counts > 0L)
    rec <- select_recommendation(
      fit$model, evaluated, sim, return_curve = TRUE
    )
    pred <- rec$pred
    rec_local <- rec$local
    rec_idx <- rec$index
    mu <- pred$mu_curve[rec_local, ]
    sd <- pred$sd_curve[rec_local, ]
    truth <- sim$true_mean[rec_idx, ]
    optimum <- sim$target_curve
    ylim <- range(mu - 1.96 * sd, mu + 1.96 * sd, truth, optimum)
    plot(sim$t, truth, type = "l", lwd = 2, col = "black",
         ylim = ylim, xlab = "t", ylab = "Functional response",
         main = method)
    polygon(
      c(sim$t, rev(sim$t)),
      c(mu - 1.96 * sd, rev(mu + 1.96 * sd)),
      col = grDevices::adjustcolor("#377EB8", alpha.f = 0.18),
      border = NA
    )
    lines(sim$t, mu, col = "#E41A1C", lwd = 2, lty = 2)
    lines(sim$t, truth, col = "black", lwd = 2)
    lines(sim$t, optimum, col = "#238B45", lwd = 2, lty = 3)
    legend(
      "topleft",
      legend = c("True at recommendation", "Posterior mean",
                 "Feasible target"),
      col = c("black", "#E41A1C", "#238B45"),
      lty = c(1, 2, 3), lwd = 2, bty = "n", cex = 0.70
    )
    grid(col = "grey88")
  }
  invisible(NULL)
}

plot_budget_performance <- function(budget_results) {
  if (is.null(budget_results)) return(invisible(NULL))
  oldpar <- par(no.readonly = TRUE)
  on.exit(par(oldpar), add = TRUE)
  set_plot_style(
    mfrow = c(1, 3), mar = c(4.1, 4.3, 1.8, 0.5),
    oma = c(0.2, 0.2, 0.2, 0.2)
  )
  methods <- c("GPFR-EQI", "WDEI")
  cols <- c("GPFR-EQI" = "#4DAF4A", "WDEI" = "#1F78B4")
  pchs <- c("GPFR-EQI" = 17, "WDEI" = 19)
  
  draw_methods <- function(metric, ylab, label) {
    budgets <- sort(unique(budget_results$Budget))
    summaries <- lapply(methods, function(method) {
      vapply(budgets, function(value) {
        budget_results[
          budget_results$Budget == value &
            budget_results$Method == method, metric
        ][1]
      }, numeric(1))
    })
    ylim <- range(unlist(summaries), finite = TRUE)
    plot(budgets, summaries[[1]], type = "n", ylim = ylim,
         xlab = "Total budget N", ylab = ylab)
    for (i in seq_along(methods)) {
      s <- summaries[[i]]
      lines(budgets, s, type = "b", pch = pchs[methods[i]],
            lwd = 2, col = cols[methods[i]])
    }
    grid(col = "grey88")
    mtext(label, side = 3, adj = 0, line = 0.25, font = 2)
  }
  
  draw_methods("Global_RMSE", "Global RMSE", "(a)")
  draw_methods("POG", "Recommendation POG", "(b)")
  
  wdei <- budget_results[budget_results$Method == "WDEI", , drop = FALSE]
  budgets <- sort(unique(wdei$Budget))
  rec <- vapply(budgets, function(value) {
    wdei[wdei$Budget == value, "POG"][1]
  }, numeric(1))
  oracle <- vapply(budgets, function(value) {
    wdei[wdei$Budget == value, "Oracle_POG"][1]
  }, numeric(1))
  ylim <- range(rec, oracle, finite = TRUE)
  plot(budgets, rec, type = "n", ylim = ylim,
       xlab = "Total budget N", ylab = "WDEI POG")
  lines(budgets, rec, type = "b", pch = 19, lwd = 2,
        col = "#1F78B4")
  lines(budgets, oracle, type = "b", pch = 17, lwd = 2,
        col = "#6A3D9A")
  grid(col = "grey88")
  legend("topright", legend = c("Recommendation", "Best evaluated (oracle)"),
         col = c("#1F78B4", "#6A3D9A"), pch = c(19, 17), lwd = 2,
         bty = "n", cex = 0.78)
  mtext("(c)", side = 3, adj = 0, line = 0.25, font = 2)
  invisible(NULL)
}

plot_sensitivity_results <- function(design_budget_results, alpha_results) {
  if (is.null(design_budget_results) || is.null(alpha_results)) {
    return(invisible(NULL))
  }
  oldpar <- par(no.readonly = TRUE)
  on.exit(par(oldpar), add = TRUE)
  set_plot_style(
    mfrow = c(2, 3), mar = c(4.0, 4.2, 1.8, 0.5),
    oma = c(0.2, 0.2, 0.2, 0.2)
  )
  
  panel <- function(df, setting, metric, xlab, ylab, label, col) {
    x <- sort(unique(df[[setting]]))
    s <- vapply(x, function(value) {
      df[df[[setting]] == value, metric][1]
    }, numeric(1))
    ylim <- range(s, finite = TRUE)
    plot(
      x, s, type = "b", pch = 19, lwd = 1.8,
      col = col, ylim = ylim, xlab = xlab, ylab = ylab
    )
    grid(col = "grey88")
    box()
    mtext(label, side = 3, adj = 0, line = 0.25,
          font = 2, cex = 1.00)
  }
  
  budget_results <- design_budget_results[
    design_budget_results$Analysis == "Total budget N" &
      design_budget_results$Method == "WDEI", , drop = FALSE
  ]
  initial_results <- design_budget_results[
    design_budget_results$Analysis == "Initial design M0" &
      design_budget_results$Method == "WDEI", , drop = FALSE
  ]
  
  panel(budget_results, "Setting", "Global_RMSE",
        "Total budget N", "Global RMSE", "(a)", "#1F78B4")
  panel(initial_results, "Setting", "Global_RMSE",
        expression(M[0]), "Global RMSE", "(b)", "#1F78B4")
  panel(alpha_results, "Alpha", "Global_RMSE",
        expression(alpha), "Global RMSE", "(c)", "#1F78B4")
  panel(budget_results, "Setting", "POG",
        "Total budget N", "POG", "(d)", "#E31A1C")
  panel(initial_results, "Setting", "POG",
        expression(M[0]), "POG", "(e)", "#E31A1C")
  panel(alpha_results, "Alpha", "POG",
        expression(alpha), "POG", "(f)", "#E31A1C")
  invisible(NULL)
}

build_table1 <- function(raw) {
  order <- MAIN_METHODS
  cell <- function(x, digits = 4L) {
    sprintf(paste0("%.", digits, "f"), x[1])
  }
  do.call(rbind, lapply(order, function(method) {
    z <- raw[raw$Method == method, , drop = FALSE]
    data.frame(
      Method = method,
      `Global RMSE` = cell(z$Global_RMSE),
      `Integrated variance` = cell(z$Integrated_Variance),
      POG = cell(z$POG),
      `Optimal RMSE` = cell(z$Opt_RMSE),
      `Unique inputs` = cell(z$Unique_Inputs, 2L),
      `Repeated trials` = cell(z$Replicates, 2L),
      check.names = FALSE
    )
  }))
}

build_table2 <- function(raw) {
  cell <- function(x, digits = 4L) {
    sprintf(paste0("%.", digits, "f"), x[1])
  }
  do.call(rbind, lapply(ABLATION_METHODS, function(method) {
    z <- raw[raw$Method == method, , drop = FALSE]
    data.frame(
      Method = method,
      `Global RMSE` = cell(z$Global_RMSE),
      `Integrated variance` = cell(z$Integrated_Variance),
      POG = cell(z$POG),
      `Optimal RMSE` = cell(z$Opt_RMSE),
      `Repeated trials` = cell(z$Replicates, 2L),
      `Useful repeats` = cell(z$Useful_Repeats, 2L),
      `Redundant repeats` = cell(z$Redundant_Repeats, 2L),
      `Activation rate` = cell(z$Activation_Rate, 3L),
      check.names = FALSE
    )
  }))
}

plot_objective_and_convergence <- function(main_results) {
  stopifnot(!is.null(main_results), !is.null(main_results$first))
  sim <- main_results$first$sim
  oldpar <- par(no.readonly = TRUE)
  on.exit(par(oldpar), add = TRUE)
  set_plot_style(
    mfrow = c(1, 2), mar = c(4.2, 4.5, 1.7, 0.7),
    oma = c(0.2, 0.2, 0.1, 0.1)
  )
  
  corner_targets <- rbind(c(-1, 6), c(-1, 8), c(1, 6), c(1, 8))
  corner_idx <- apply(corner_targets, 1, function(v) {
    which.min(rowSums((sweep(sim$U, 2, v, "-"))^2))
  })
  cols <- c("#377EB8", "#4DAF4A", "#984EA3", "#FF7F00")
  yrange <- range(sim$true_mean[corner_idx, ], sim$target_curve)
  plot(
    sim$t, sim$true_mean[corner_idx[1], ], type = "l",
    lwd = 1.5, col = cols[1], ylim = yrange,
    xlab = "Functional index t", ylab = "Latent mean response"
  )
  for (i in 2:4) {
    lines(sim$t, sim$true_mean[corner_idx[i], ],
          lwd = 1.5, col = cols[i])
  }
  lines(sim$t, sim$target_curve, lwd = 2.6, col = "black")
  grid(col = "grey88")
  legend(
    "topleft",
    legend = c("u=(-1,6)", "u=(-1,8)", "u=(1,6)", "u=(1,8)",
               "Target/optimal u=(-0.2,6.8)"),
    col = c(cols, "black"), lwd = c(rep(1.5, 4), 2.6),
    bty = "n", cex = 0.72
  )
  mtext("(a)", side = 3, adj = 0, line = 0.25,
        font = 2, cex = 1.00)
  
  s <- summarize_history(
    main_results$history, metric = "RecommendationRegret"
  )
  methods <- c("GPFR-EQI", "WDEI")
  cols2 <- c("GPFR-EQI" = "#4DAF4A", "WDEI" = "#1F78B4")
  s <- s[s$Method %in% methods, , drop = FALSE]
  ylim <- range(pmax(s$Value, 0), finite = TRUE)
  plot(
    range(s$Iteration), ylim, type = "n",
    xlab = "Sequential iteration", ylab = "Feasible POG"
  )
  for (method in methods) {
    z <- s[s$Method == method, , drop = FALSE]
    z <- z[order(z$Iteration), ]
    lines(z$Iteration, z$Value, col = cols2[method], lwd = 2)
  }
  grid(col = "grey88")
  legend("topright", legend = methods, col = cols2[methods],
         lwd = 2, bty = "n", cex = 0.80)
  mtext("(b)", side = 3, adj = 0, line = 0.25,
        font = 2, cex = 1.00)
  invisible(s)
}

plot_figure4 <- function(main_history, ablation_history) {
  stopifnot(!is.null(main_history), !is.null(ablation_history))
  h <- main_history[main_history$Method == "WDEI", , drop = FALSE]
  h_ab <- ablation_history[
    ablation_history$Method %in% c("FEI+D", "WDEI"), , drop = FALSE
  ]
  oldpar <- par(no.readonly = TRUE)
  on.exit(par(oldpar), add = TRUE)
  set_plot_style(
    mfrow = c(2, 2), mar = c(4.0, 4.6, 1.7, 0.6),
    oma = c(0.2, 0.2, 0.2, 0.2)
  )
  
  draw_trace <- function(data, metric, ylab, col, label) {
    iterations <- sort(unique(h$Iteration))
    values <- vapply(iterations, function(iter) {
      data[data$Iteration == iter, metric][1]
    }, numeric(1))
    plot(
      iterations, values, type = "l", lwd = 2, col = col,
      xlab = "Sequential iteration", ylab = ylab
    )
    grid(col = "grey88")
    box()
    mtext(label, side = 3, adj = 0, line = 0.25,
          font = 2, cex = 0.95)
    invisible(values)
  }
  
  draw_trace(h, "ActiveDecay", "Activated decay locations",
             "#E31A1C", "(a)")
  
  iterations <- sort(unique(h_ab$Iteration))
  methods <- c("FEI+D", "WDEI")
  cols <- c("FEI+D" = "#6A3D9A", "WDEI" = "#1F78B4")
  traces <- lapply(methods, function(method) {
    z <- h_ab[h_ab$Method == method, , drop = FALSE]
    vapply(iterations, function(iter) {
      z[z$Iteration == iter, "CumulativeRedundantRepeats"][1]
    }, numeric(1))
  })
  ylim <- range(unlist(traces), finite = TRUE)
  plot(iterations, traces[[1]], type = "n", ylim = ylim,
       xlab = "Sequential iteration", ylab = "Cumulative redundant repeats")
  for (i in seq_along(methods)) {
    lines(iterations, traces[[i]], lwd = 2, col = cols[methods[i]])
  }
  grid(col = "grey88")
  legend("topleft", legend = methods, col = cols[methods], lwd = 2,
         bty = "n", cex = 0.82)
  mtext("(b)", side = 3, adj = 0, line = 0.25, font = 2, cex = 0.95)
  
  fei_metrics <- c("MaxFEI", "MaxWeightedFEI")
  fei_labels <- c("Original FEI", "Weighted FEI")
  fei_cols <- c("#4D4D4D", "#E31A1C")
  fei_traces <- lapply(fei_metrics, function(metric) {
    vapply(iterations, function(iter) {
      h[h$Iteration == iter, metric][1]
    }, numeric(1))
  })
  ylim <- range(unlist(fei_traces), finite = TRUE)
  plot(iterations, fei_traces[[1]], type = "n", ylim = ylim,
       xlab = "Sequential iteration", ylab = "Maximum FEI")
  for (i in seq_along(fei_metrics)) {
    lines(iterations, fei_traces[[i]], lwd = 2, col = fei_cols[i],
          lty = if (i == 1L) 1 else 2)
  }
  grid(col = "grey88")
  legend("topright", legend = fei_labels, col = fei_cols,
         lty = c(1, 2), lwd = 2, bty = "n", cex = 0.82)
  mtext("(c)", side = 3, adj = 0, line = 0.25, font = 2, cex = 0.95)
  
  draw_trace(h, "IntegratedVariance", "Integrated latent variance",
             "#33A02C", "(d)")
  invisible(h)
}

plot_figure5 <- function(first_result) {
  stopifnot(!is.null(first_result), !is.null(first_result$sim))
  sim <- first_result$sim
  details <- first_result$details
  oldpar <- par(no.readonly = TRUE)
  on.exit(par(oldpar), add = TRUE)
  set_plot_style(
    mfrow = c(2, 2), mar = c(3.8, 4.2, 2.2, 0.7),
    oma = c(0.3, 0.3, 0.2, 0.2)
  )
  
  curve_data <- lapply(MAIN_METHODS, function(method) {
    fit <- details[[method]]
    evaluated <- which(fit$counts > 0L)
    rec <- select_recommendation(
      fit$model, evaluated, sim, return_curve = TRUE
    )
    pred <- rec$pred
    local <- rec$local
    idx <- rec$index
    list(
      method = method, idx = idx,
      mu = pred$mu_curve[local, ],
      sd = pred$sd_curve[local, ],
      truth = sim$true_mean[idx, ]
    )
  })
  common_ylim <- range(unlist(lapply(curve_data, function(z) {
    c(z$mu - 1.96 * z$sd, z$mu + 1.96 * z$sd,
      z$truth, sim$target_curve)
  })))
  
  for (i in seq_along(curve_data)) {
    z <- curve_data[[i]]
    u_label <- paste(sprintf("%.2f", sim$U[z$idx, ]), collapse = ", ")
    plot(
      sim$t, z$truth, type = "n", ylim = common_ylim,
      xlab = "t", ylab = "Functional response",
      main = sprintf("%s\nu=(%s)", z$method, u_label),
      cex.main = 0.90
    )
    polygon(
      c(sim$t, rev(sim$t)),
      c(z$mu - 1.96 * z$sd, rev(z$mu + 1.96 * z$sd)),
      col = adjustcolor("#7F7F7F", alpha.f = 0.20), border = NA
    )
    lines(sim$t, sim$target_curve, col = "black", lwd = 2, lty = 3)
    lines(sim$t, z$truth, col = "#377EB8", lwd = 2)
    lines(sim$t, z$mu, col = "#E41A1C", lwd = 2, lty = 2)
    grid(col = "grey88")
    box()
    legend(
      "topleft",
      legend = c("Target/optimal curve",
                 "True curve at recommendation",
                 "GPFR posterior mean", "95% latent-mean CI"),
      col = c("black", "#377EB8", "#E41A1C", NA),
      lty = c(3, 1, 2, NA), lwd = c(2, 2, 2, NA),
      fill = c(NA, NA, NA,
               adjustcolor("#7F7F7F", alpha.f = 0.20)),
      border = c(NA, NA, NA, NA), bty = "n", cex = 0.64,
      inset = 0.01
    )
    mtext(paste0("(", letters[i], ")"), side = 3, adj = 0,
          line = 0.20, font = 2, cex = 0.90)
  }
  invisible(curve_data)
}

# ==============================================================================
# 7. Main comparison
# ==============================================================================

MAIN_RESULTS <- NULL
TABLE1_MAIN <- NULL

if (RUN_MAIN) {
  cat("\n\n######################################################################\n")
  cat("Starting the main comparison: Initial LHS, One-shot LHS, GPFR-EQI, and WDEI\n")
  cat("######################################################################\n")
  
  MAIN_RESULTS <- run_experiment(
    n_rep = N_REP_MAIN,
    methods = MAIN_METHODS,
    cfg = CFG,
    keep_first = TRUE,
    label = "Main comparison"
  )
  TABLE1_MAIN <- build_table1(MAIN_RESULTS$raw)
  cat("\n\n================ Table 1: main results, single-run values ================\n")
  print(TABLE1_MAIN, row.names = FALSE, right = FALSE)
  cat(sprintf("\nTotal runtime for the main comparison: %.2f seconds\n", MAIN_RESULTS$seconds))
  cat(sprintf(
    "True optimal attainable input in the main scenario: u0=%.3f, u1=%.3f, J*=%.6f\n",
    MAIN_RESULTS$first$sim$U[
      MAIN_RESULTS$first$sim$true_best_idx, 1
    ],
    MAIN_RESULTS$first$sim$U[
      MAIN_RESULTS$first$sim$true_best_idx, 2
    ],
    MAIN_RESULTS$first$sim$true_J[
      MAIN_RESULTS$first$sim$true_best_idx
    ]
  ))
  cat("The recommended-profile plot uses the single run (Rep = 1).\n")
  cat(sprintf(
    paste0("Process variance in the main example = %.4f; ",
           "mean input-signal variance over the candidate set = %.4f; ",
           "ratio = %.3f.\n"),
    CFG$process_var,
    MAIN_RESULTS$first$sim$mean_input_signal_variance,
    CFG$process_var /
      MAIN_RESULTS$first$sim$mean_input_signal_variance
  ))
  plot_figure5(MAIN_RESULTS$first)
}

# ==============================================================================
# 8. WDEI ablation study
# ==============================================================================

ABLATION_RESULTS <- NULL
TABLE2_ABLATION <- NULL

if (RUN_ABLATION) {
  cat("\n\n######################################################################\n")
  cat("Starting the ablation study: GPFR-FEI, FEI+W, FEI+D, and WDEI\n")
  cat("######################################################################\n")
  
  ABLATION_RESULTS <- run_experiment(
    n_rep = N_REP_ABLATION,
    methods = ABLATION_METHODS,
    cfg = CFG,
    keep_first = FALSE,
    label = "Ablation study"
  )
  
  ablation_metrics <- c(
    "Global_RMSE", "Opt_RMSE", "POG",
    "Integrated_Variance", "Unique_Inputs", "Replicates",
    "Effective_Repeat_Rate"
  )
  cat("\n================ WDEI ablation results ================\n")
  print(
    summarize_metrics(ABLATION_RESULTS$raw, ablation_metrics),
    row.names = FALSE
  )
  cat(sprintf("\nTotal runtime for the ablation study: %.2f seconds\n",
              ABLATION_RESULTS$seconds))
  TABLE2_ABLATION <- build_table2(ABLATION_RESULTS$raw)
  cat("\n================ Table 2: ablation results, single-run values ================\n")
  print(TABLE2_ABLATION, row.names = FALSE, right = FALSE)
  
  if (!is.null(MAIN_RESULTS)) {
    plot_figure4(MAIN_RESULTS$history, ABLATION_RESULTS$history)
  }
}

# ==============================================================================
# 9. Initial-design and total-budget analyses
# ==============================================================================

DESIGN_BUDGET_RESULTS <- NULL

if (RUN_DESIGN_BUDGET) {
  cat("\n\n######################################################################\n")
  cat("Starting the design and budget analysis: N = 30, 40, 60; M0 = 10, 20, 30\n")
  cat("######################################################################\n")
  
  sequential_methods <- c("GPFR-EQI", "WDEI")
  analysis_rows <- list()
  counter <- 1L
  
  run_or_reuse_default <- function(cfg_i, label) {
    is_default <- cfg_i$M_init == CFG$M_init &&
      cfg_i$N_total == CFG$N_total
    if (is_default && !is.null(MAIN_RESULTS)) {
      return(MAIN_RESULTS$raw[
        MAIN_RESULTS$raw$Method %in% sequential_methods, , drop = FALSE
      ])
    }
    run_experiment(
      n_rep = N_REP_DESIGN_BUDGET,
      methods = sequential_methods,
      cfg = cfg_i, keep_first = FALSE, label = label
    )$raw
  }
  
  for (value in c(30, 40, 60)) {
    cfg_i <- CFG
    cfg_i$N_total <- value
    cfg_i$verbose <- FALSE
    temp <- run_or_reuse_default(cfg_i, paste0("N=", value))
    temp$Analysis <- "Total budget N"
    temp$Setting <- value
    temp$Budget <- value
    temp$Initial_Size <- cfg_i$M_init
    analysis_rows[[counter]] <- temp
    counter <- counter + 1L
  }
  
  for (value in c(10, 20, 30)) {
    cfg_i <- CFG
    cfg_i$M_init <- value
    cfg_i$verbose <- FALSE
    temp <- run_or_reuse_default(cfg_i, paste0("M0=", value))
    temp$Analysis <- "Initial design M0"
    temp$Setting <- value
    temp$Budget <- cfg_i$N_total
    temp$Initial_Size <- value
    analysis_rows[[counter]] <- temp
    counter <- counter + 1L
  }
  
  DESIGN_BUDGET_RESULTS <- do.call(rbind, analysis_rows)
  cat("\n================ Total-budget analysis ================\n")
  print(summarize_by_setting(
    DESIGN_BUDGET_RESULTS[
      DESIGN_BUDGET_RESULTS$Analysis == "Total budget N", , drop = FALSE
    ], "Setting",
    metrics = c("Global_RMSE", "Integrated_Variance", "POG",
                "Opt_RMSE", "Replicates"),
    method_column = "Method"
  ), row.names = FALSE)
  cat("\n================ Initial-design-size analysis ================\n")
  print(summarize_by_setting(
    DESIGN_BUDGET_RESULTS[
      DESIGN_BUDGET_RESULTS$Analysis == "Initial design M0", , drop = FALSE
    ], "Setting",
    metrics = c("Global_RMSE", "Integrated_Variance", "POG",
                "Opt_RMSE", "Replicates"),
    method_column = "Method"
  ), row.names = FALSE)
}

# ==============================================================================
# 10. Parameter sensitivity analyses
# ==============================================================================

SENSITIVITY_ALPHA_RESULTS <- NULL
SENSITIVITY_KAPPA_RESULTS <- NULL
SENSITIVITY_ETA_RESULTS <- NULL
SENSITIVITY_RHO_RESULTS <- NULL

run_wdei_parameter_grid <- function(values, cfg_field, output_name,
                                    label_prefix) {
  out <- vector("list", length(values))
  for (i in seq_along(values)) {
    cfg_i <- CFG
    cfg_i[[cfg_field]] <- values[i]
    cfg_i$verbose <- FALSE
    ans <- run_experiment(
      n_rep = N_REP_SENSITIVITY,
      methods = "WDEI", cfg = cfg_i,
      keep_first = FALSE,
      label = paste0(label_prefix, "=", values[i])
    )
    temp <- ans$raw
    temp[[output_name]] <- values[i]
    out[[i]] <- temp
  }
  do.call(rbind, out)
}

if (RUN_ALPHA_SENSITIVITY) {
  cat("\n\n######################################################################\n")
  cat("Starting the alpha trade-off analysis\n")
  cat("######################################################################\n")
  
  alpha_values <- c(0, 0.25, 0.50, 0.75, 1.00)
  SENSITIVITY_ALPHA_RESULTS <- run_wdei_parameter_grid(
    alpha_values, "alpha", "Alpha", "alpha"
  )
  
  cat("\n================ Alpha sensitivity ================\n")
  print(compact_setting_table(SENSITIVITY_ALPHA_RESULTS, "Alpha"),
        row.names = FALSE, right = FALSE)
}

if (RUN_REGULATION_SENSITIVITY) {
  cat("\n\n######################################################################\n")
  cat("Starting the robustness analysis for kappa, eta/eta0, and rho\n")
  cat("######################################################################\n")
  
  kappa_values <- c(0.50, 1.00, 2.00)
  eta_scale_values <- c(0.50, 1.00, 2.00)
  rho_values <- c(0.25, 0.50, 0.75)
  
  SENSITIVITY_KAPPA_RESULTS <- run_wdei_parameter_grid(
    kappa_values, "kappa", "Kappa", "kappa"
  )
  SENSITIVITY_ETA_RESULTS <- run_wdei_parameter_grid(
    eta_scale_values, "eta_multiplier", "EtaScale", "eta/eta0"
  )
  SENSITIVITY_RHO_RESULTS <- run_wdei_parameter_grid(
    rho_values, "rho", "Rho", "rho"
  )
  
  cat("\n================ Kappa sensitivity ================\n")
  print(compact_setting_table(
    SENSITIVITY_KAPPA_RESULTS, "Kappa",
    metrics = c("Global_RMSE", "Opt_RMSE", "POG",
                "Unique_Inputs", "Replicates")
  ), row.names = FALSE, right = FALSE)
  cat("\n================ Eta sensitivity ================\n")
  print(compact_setting_table(
    SENSITIVITY_ETA_RESULTS, "EtaScale",
    metrics = c("Global_RMSE", "POG",
                "Unique_Inputs", "Replicates")
  ), row.names = FALSE, right = FALSE)
  cat("\n================ Rho sensitivity ================\n")
  print(compact_setting_table(
    SENSITIVITY_RHO_RESULTS, "Rho",
    metrics = c("Global_RMSE", "POG",
                "Unique_Inputs", "Replicates")
  ), row.names = FALSE, right = FALSE)
}

if (RUN_DESIGN_BUDGET && RUN_ALPHA_SENSITIVITY) {
  plot_sensitivity_results(DESIGN_BUDGET_RESULTS,
                           SENSITIVITY_ALPHA_RESULTS)
}

# ==============================================================================
# 11. Computational scalability analysis
# ==============================================================================

SCALABILITY_RESULTS <- NULL
SCALABILITY_SUMMARY <- NULL

if (RUN_SCALABILITY) {
  cat("\n\n######################################################################\n")
  cat("Starting the computational scalability analysis\n")
  cat("######################################################################\n")
  
  scale_cases <- data.frame(
    Scale_Factor = c(
      rep("Curves_M", 3),
      rep("Time_Points_n", 3),
      rep("Candidates_J", 3)
    ),
    Scale_Level = c(20, 40, 80, 21, 41, 81, 100, 225, 441),
    M = c(20, 40, 80, 40, 40, 40, 40, 40, 40),
    n_time = c(41, 41, 41, 21, 41, 81, 41, 41, 41),
    grid_each = c(15, 15, 15, 15, 15, 15, 10, 15, 21),
    stringsAsFactors = FALSE
  )
  scale_rows <- list()
  scale_counter <- 1L
  
  for (case in seq_len(nrow(scale_cases))) {
    cfg_i <- CFG
    cfg_i$M_init <- scale_cases$M[case]
    cfg_i$N_total <- scale_cases$M[case]
    cfg_i$n_time <- scale_cases$n_time[case]
    cfg_i$n_basis <- min(12, max(7, round(cfg_i$n_time / 5)))
    cfg_i$grid_each <- scale_cases$grid_each[case]
    cfg_i$diversity_reference <- cfg_i$grid_each^2
    cfg_i$tune_hyper <- FALSE
    cfg_i$verbose <- FALSE
    
    for (r in seq_len(N_REP_SCALABILITY)) {
      sim <- make_simulation(case * 1000L + r + 5000L, cfg_i)
      lhs <- lhs_unit(
        cfg_i$M_init, 2,
        seed = cfg_i$seed + case * 1000L + r
      )
      idx <- match_lhs_to_candidates(lhs, sim$U_unit)
      eta_scale <- derive_eta_from_initial_design(
        sim$U_unit, idx, cfg_i
      )
      sim$cfg$eta <- eta_scale$eta
      sim$eta_scale <- eta_scale
      obs <- observe_indices(sim, idx)
      hyper <- list(
        ell_u = rep(cfg_i$kernel_ell_u, ncol(sim$U_unit)),
        ell_x = cfg_i$kernel_ell_x
      )
      
      start_fit <- proc.time()[3]
      model <- fit_functional_gp(
        obs$Y, idx, obs$X_obs_scaled, sim, hyper
      )
      fit_seconds <- proc.time()[3] - start_fit
      
      start_pred <- proc.time()[3]
      invisible(predict_functional_gp(
        model, seq_len(nrow(sim$U)), return_curve = TRUE
      ))
      predict_seconds <- proc.time()[3] - start_pred
      
      counts <- tabulate(idx, nbins = nrow(sim$U))
      start_acq <- proc.time()[3]
      acq_object <- functional_acquisition(
        "WDEI", model, counts, sim
      )
      acquisition_seconds <- proc.time()[3] - start_acq
      
      scale_rows[[scale_counter]] <- data.frame(
        Rep = r,
        Scale_Factor = scale_cases$Scale_Factor[case],
        Scale_Level = scale_cases$Scale_Level[case],
        Curves_M = cfg_i$M_init,
        Time_Points_n = cfg_i$n_time,
        Basis_H = cfg_i$n_basis,
        Candidates_J = nrow(sim$U),
        Fit_Seconds = fit_seconds,
        Predict_Seconds = predict_seconds,
        WDEI_Acquisition_Seconds = acquisition_seconds,
        Total_Iteration_Seconds = fit_seconds + acquisition_seconds,
        Model_MB = as.numeric(object.size(model)) / 1024^2,
        Acquisition_Object_MB =
          as.numeric(object.size(acq_object)) / 1024^2
      )
      scale_counter <- scale_counter + 1L
    }
    cat(sprintf(
      "%s=%g; M=%d, n=%d, J=%d: completed %d timing replications\n",
      scale_cases$Scale_Factor[case], scale_cases$Scale_Level[case],
      cfg_i$M_init, cfg_i$n_time, cfg_i$grid_each^2,
      N_REP_SCALABILITY
    ))
  }
  SCALABILITY_RESULTS <- do.call(rbind, scale_rows)
  cat("\n================ Computational scalability results ================\n")
  SCALABILITY_SUMMARY <- aggregate(
    cbind(Fit_Seconds, Predict_Seconds, WDEI_Acquisition_Seconds,
          Total_Iteration_Seconds, Model_MB, Acquisition_Object_MB) ~
      Scale_Factor + Scale_Level + Curves_M + Time_Points_n +
      Basis_H + Candidates_J,
    data = SCALABILITY_RESULTS,
    FUN = mean
  )
  print(SCALABILITY_SUMMARY, row.names = FALSE)
}

# ==============================================================================
# 12. Six-dimensional nonlinear example in Appendix B
# ==============================================================================

plot_appendix_b1 <- function(first_result) {
  sim <- first_result$sim
  oldpar <- par(no.readonly = TRUE)
  on.exit(par(oldpar), add = TRUE)
  set_plot_style(mfrow = c(1, 2), mar = c(4.2, 4.5, 1.7, 0.7))
  
  idx <- unique(round(seq(1, nrow(sim$U), length.out = 8)))
  matplot(
    sim$t, t(sim$true_mean[idx, , drop = FALSE]),
    type = "l", lty = 1, lwd = 1.5,
    col = grDevices::hcl.colors(length(idx), "Dark 3"),
    xlab = "Functional index t", ylab = "Latent response"
  )
  grid(col = "grey88")
  mtext("(a)", side = 3, adj = 0, line = 0.25,
        font = 2, cex = 1.00)
  
  g <- seq(0, 1, length.out = 51)
  z <- outer(g, g, Vectorize(function(u1, u2) {
    u <- c(u1, u2, rep(0.5, 4))
    sum(u^2) - 0.8 * cos(2 * pi * u1 * u2)
  }))
  contour(
    g, g, z, nlevels = 14, drawlabels = TRUE,
    xlab = expression(u[1]), ylab = expression(u[2]),
    col = "#4D4D4D"
  )
  mtext("(b)", side = 3, adj = 0, line = 0.25,
        font = 2, cex = 1.00)
  invisible(NULL)
}

plot_appendix_b2 <- function(first_result) {
  sim <- first_result$sim
  fit <- first_result$details[["WDEI"]]
  oldpar <- par(no.readonly = TRUE)
  on.exit(par(oldpar), add = TRUE)
  set_plot_style(mfrow = c(1, 2), mar = c(4.2, 4.5, 1.7, 0.7))
  
  h <- fit$history
  plot(
    h$Iteration, h$MaxAcquisition, type = "l", lwd = 2,
    col = "#1F78B4", xlab = "Sequential iteration",
    ylab = "Maximum WDEI"
  )
  grid(col = "grey88")
  mtext("(a)", side = 3, adj = 0, line = 0.25,
        font = 2, cex = 1.00)
  
  evaluated <- which(fit$counts > 0L)
  rec <- select_recommendation(
    fit$model, evaluated, sim, return_curve = TRUE
  )
  pred <- rec$pred
  local <- rec$local
  idx <- rec$index
  mu <- pred$mu_curve[local, ]
  sd <- pred$sd_curve[local, ]
  truth <- sim$true_mean[idx, ]
  ylim <- range(mu - 1.96 * sd, mu + 1.96 * sd,
                truth, sim$target_curve)
  plot(
    sim$t, truth, type = "n", ylim = ylim,
    xlab = "Functional index t", ylab = "Functional response"
  )
  polygon(
    c(sim$t, rev(sim$t)),
    c(mu - 1.96 * sd, rev(mu + 1.96 * sd)),
    col = adjustcolor("#7F7F7F", alpha.f = 0.20), border = NA
  )
  lines(sim$t, sim$target_curve, col = "black", lwd = 2, lty = 3)
  lines(sim$t, truth, col = "#377EB8", lwd = 2)
  lines(sim$t, mu, col = "#E41A1C", lwd = 2, lty = 2)
  grid(col = "grey88")
  legend(
    "topleft",
    legend = c("Target", "True at recommendation", "Posterior mean"),
    col = c("black", "#377EB8", "#E41A1C"),
    lty = c(3, 1, 2), lwd = 2, bty = "n", cex = 0.78
  )
  mtext("(b)", side = 3, adj = 0, line = 0.25,
        font = 2, cex = 1.00)
  invisible(NULL)
}

APPENDIX6D_RESULTS <- NULL
TABLE_B1 <- NULL

if (RUN_APPENDIX_6D) {
  cat("\n\n######################################################################\n")
  cat("Starting the six-dimensional nonlinear example in Appendix B\n")
  cat("######################################################################\n")
  
  CFG_B <- CFG
  CFG_B$case_id <- "appendix6d"
  CFG_B$scenario <- "appendix6d"
  CFG_B$surrogate <- "flexible"
  CFG_B$t_min <- 0
  CFG_B$t_max <- 2
  CFG_B$n_time <- 41
  CFG_B$n_basis <- 10
  CFG_B$input_lower <- rep(0, 6)
  CFG_B$input_upper <- rep(1, 6)
  CFG_B$target_u <- c(0.25, 0.40, 0.50, 0.30, 0.50, 0.20)
  CFG_B$candidate_size <- 600
  CFG_B$M_init <- 30
  CFG_B$N_total <- 90
  CFG_B$process_var <- 0
  CFG_B$noise_sd <- 0.20
  CFG_B$diversity_reference <- 100
  CFG_B$kernel_ell_u <- 0.30
  CFG_B$kernel_ell_x <- 1
  CFG_B$objective_mc <- 256
  CFG_B$verbose <- TRUE
  
  n_rep_b <- 1
  APPENDIX6D_RESULTS <- run_experiment(
    n_rep = n_rep_b,
    methods = MAIN_METHODS,
    cfg = CFG_B,
    keep_first = TRUE,
    label = "Appendix B six-dimensional example"
  )
  TABLE_B1 <- build_table1(APPENDIX6D_RESULTS$raw)
  cat("\n================ Table B1: six-dimensional example, single-run values ================\n")
  print(TABLE_B1, row.names = FALSE, right = FALSE)
  plot_appendix_b1(APPENDIX6D_RESULTS$first)
  plot_appendix_b2(APPENDIX6D_RESULTS$first)
}

# ==============================================================================
# 13. Export of tables and figures
# ==============================================================================

open_pdf_device <- function(path, width, height) {
  if (capabilities("cairo")) {
    grDevices::cairo_pdf(
      filename = path, width = width, height = height,
      family = PDF_FAMILY, onefile = TRUE
    )
  } else {
    grDevices::pdf(
      file = path, width = width, height = height,
      family = PDF_FAMILY, onefile = TRUE
    )
  }
}

save_figure_pair <- function(name, plot_fun, width, height) {
  pdf_path <- file.path(OUTPUT_DIR, paste0(name, ".pdf"))
  tif_path <- file.path(OUTPUT_DIR, paste0(name, ".tiff"))
  
  open_pdf_device(pdf_path, width, height)
  plot_fun()
  invisible(dev.off())
  
  grDevices::tiff(
    filename = tif_path, width = width, height = height,
    units = "in", res = 600, compression = "lzw"
  )
  plot_fun()
  invisible(dev.off())
}

if (SAVE_PAPER_OUTPUTS) {
  if (!dir.exists(OUTPUT_DIR)) dir.create(OUTPUT_DIR, recursive = TRUE)
  
  if (!is.null(TABLE1_MAIN)) {
    write.csv(
      TABLE1_MAIN, file.path(OUTPUT_DIR, "Table1_Main_Results.csv"),
      row.names = FALSE, fileEncoding = "UTF-8"
    )
    write.csv(
      MAIN_RESULTS$raw, file.path(OUTPUT_DIR, "Main_Raw.csv"),
      row.names = FALSE, fileEncoding = "UTF-8"
    )
    write.csv(
      MAIN_RESULTS$history, file.path(OUTPUT_DIR, "Main_History.csv"),
      row.names = FALSE, fileEncoding = "UTF-8"
    )
  }
  if (!is.null(TABLE2_ABLATION)) {
    write.csv(
      TABLE2_ABLATION, file.path(OUTPUT_DIR, "Table2_Ablation.csv"),
      row.names = FALSE, fileEncoding = "UTF-8"
    )
    write.csv(
      ABLATION_RESULTS$raw, file.path(OUTPUT_DIR, "Ablation_Raw.csv"),
      row.names = FALSE, fileEncoding = "UTF-8"
    )
  }
  
  if (!is.null(DESIGN_BUDGET_RESULTS)) {
    write.csv(
      DESIGN_BUDGET_RESULTS,
      file.path(OUTPUT_DIR, "Design_Budget_Raw.csv"),
      row.names = FALSE, fileEncoding = "UTF-8"
    )
    for (analysis_name in unique(DESIGN_BUDGET_RESULTS$Analysis)) {
      z <- DESIGN_BUDGET_RESULTS[
        DESIGN_BUDGET_RESULTS$Analysis == analysis_name, , drop = FALSE
      ]
      filename <- if (analysis_name == "Total budget N") {
        "Budget_N_Summary.csv"
      } else {
        "Initial_Design_M0_Summary.csv"
      }
      write.csv(
        summarize_by_setting(
          z, "Setting",
          metrics = c("Global_RMSE", "Integrated_Variance", "POG",
                      "Oracle_POG", "Opt_RMSE", "Replicates"),
          method_column = "Method"
        ),
        file.path(OUTPUT_DIR, filename),
        row.names = FALSE, fileEncoding = "UTF-8"
      )
    }
    save_figure_pair(
      "Figure4_Budget_Dependent_Performance",
      function() plot_budget_performance(
        DESIGN_BUDGET_RESULTS[
          DESIGN_BUDGET_RESULTS$Analysis == "Total budget N", , drop = FALSE
        ]
      ), 10.2, 3.6
    )
  }
  
  if (!is.null(MAIN_RESULTS) && !is.null(ABLATION_RESULTS)) {
    save_figure_pair(
      "Figure5_WDEI_Mechanism",
      function() plot_figure4(
        MAIN_RESULTS$history, ABLATION_RESULTS$history
      ), 7.2, 5.4
    )
  }
  if (!is.null(MAIN_RESULTS)) {
    save_figure_pair(
      "FigureS1_Recommended_Curves",
      function() plot_figure5(MAIN_RESULTS$first), 8.2, 6.2
    )
  }
  
  if (!is.null(DESIGN_BUDGET_RESULTS) &&
      !is.null(SENSITIVITY_ALPHA_RESULTS)) {
    save_figure_pair(
      "Figure6_Core_Sensitivity",
      function() plot_sensitivity_results(
        DESIGN_BUDGET_RESULTS, SENSITIVITY_ALPHA_RESULTS
      ),
      10.2, 6.2
    )
  }
  
  sensitivity_objects <- list(
    Alpha = list(data = SENSITIVITY_ALPHA_RESULTS, setting = "Alpha"),
    Kappa = list(data = SENSITIVITY_KAPPA_RESULTS, setting = "Kappa"),
    EtaScale = list(data = SENSITIVITY_ETA_RESULTS,
                    setting = "EtaScale"),
    Rho = list(data = SENSITIVITY_RHO_RESULTS, setting = "Rho")
  )
  for (nm in names(sensitivity_objects)) {
    item <- sensitivity_objects[[nm]]
    obj <- item$data
    if (!is.null(obj)) {
      write.csv(
        obj, file.path(OUTPUT_DIR, paste0("Sensitivity_", nm, "_Raw.csv")),
        row.names = FALSE, fileEncoding = "UTF-8"
      )
      write.csv(
        compact_setting_table(obj, item$setting),
        file.path(OUTPUT_DIR,
                  paste0("Sensitivity_", nm, "_Summary.csv")),
        row.names = FALSE, fileEncoding = "UTF-8"
      )
    }
  }
  
  if (!is.null(SCALABILITY_RESULTS)) {
    write.csv(
      SCALABILITY_RESULTS, file.path(OUTPUT_DIR, "Scalability_Raw.csv"),
      row.names = FALSE, fileEncoding = "UTF-8"
    )
    write.csv(
      SCALABILITY_SUMMARY,
      file.path(OUTPUT_DIR, "Scalability_Summary.csv"),
      row.names = FALSE, fileEncoding = "UTF-8"
    )
  }
  
  if (!is.null(APPENDIX6D_RESULTS)) {
    write.csv(
      TABLE_B1, file.path(OUTPUT_DIR, "TableB1_Appendix6D.csv"),
      row.names = FALSE, fileEncoding = "UTF-8"
    )
    save_figure_pair(
      "FigureB1_Appendix6D_Characteristics",
      function() plot_appendix_b1(APPENDIX6D_RESULTS$first), 7.2, 3.6
    )
    save_figure_pair(
      "FigureB2_Appendix6D_WDEI",
      function() plot_appendix_b2(APPENDIX6D_RESULTS$first), 7.2, 3.6
    )
  }
}
