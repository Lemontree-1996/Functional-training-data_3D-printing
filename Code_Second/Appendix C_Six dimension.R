# ==============================================================================
# Appendix C.R
# Six-dimensional nonlinear functional-response example in Appendix C
# ==============================================================================

rm(list = ls())
options(stringsAsFactors = FALSE, scipen = 6, digits = 6)

# ==============================================================================
# 1. Experimental settings
# ==============================================================================

N_REP <- 20L
SAVE_OUTPUTS <- TRUE
OUTPUT_DIR <- "Appendix_C_1_outputs"

CFG <- list(
  seed = 20260808L,
  d = 6L,
  n_time = 41L,
  t_min = 0,
  t_max = 2,
  candidate_size = 600L,
  input_lower = rep(0, 6),
  input_upper = rep(1, 6),
  target_u = c(0.25, 0.40, 0.50, 0.30, 0.50, 0.20),
  M_init = 30L,
  N_total = 90L,
  n_basis = 10L,
  spline_degree = 3L,
  basis_penalty = 1e-6,
  noise_sd = 0.20,
  mean_ridge_linear = 1e-8,
  mean_ridge_nonlinear = 1e-3,
  kernel_ell_u = 0.30,
  signal_floor_ratio = 0.05,
  objective_mc = 256L,
  alpha = 0.50,
  rho = 0.50,
  kappa = 1.00,
  decay_at_r0 = 0.05,
  eta_multiplier = 1.00,
  eqi_beta = 0.90,
  diversity_reference = 100L,
  jitter = 1e-9,
  utility_epsilon = 1e-12,
  verbose = TRUE
)

SEQUENTIAL_METHODS <- c("GPFR-EQI", "WDEI")
MEAN_SPECIFICATIONS <- c("Nonlinear", "Linear")

PLOT_FAMILY <- "serif"
PDF_FAMILY <- "serif"
if (.Platform$OS.type == "windows") {
  windowsFonts(TNR = windowsFont("Times New Roman"))
  PLOT_FAMILY <- "TNR"
  PDF_FAMILY <- "Times New Roman"
}

set_plot_style <- function(mfrow = c(1, 1), mar = c(4.4, 4.8, 2.3, 0.8)) {
  par(
    family = PLOT_FAMILY, mfrow = mfrow, mar = mar,
    mgp = c(2.65, 0.75, 0), tcl = -0.25,
    cex.axis = 0.98, cex.lab = 1.08, cex.main = 1.05,
    las = 1, bty = "o", xaxs = "r", yaxs = "r"
  )
}

# ==============================================================================
# 2. Numerical utilities
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
  if (n > 2L) w[2:(n - 1)] <- (t[3:n] - t[1:(n - 2)]) / 2
  w
}

safe_chol <- function(K, jitter = 1e-9, max_try = 8L) {
  K <- (K + t(K)) / 2
  for (i in 0:max_try) {
    ans <- try(chol(K + diag(jitter * 10^i, nrow(K))), silent = TRUE)
    if (!inherits(ans, "try-error")) return(ans)
  }
  stop("Cholesky decomposition failed.")
}

chol_solve <- function(U, b) {
  backsolve(U, forwardsolve(t(U), b))
}

lhs_unit <- function(n, d, seed = NULL) {
  draw <- function() {
    ans <- matrix(NA_real_, nrow = n, ncol = d)
    for (j in seq_len(d)) ans[, j] <- (sample.int(n) - runif(n)) / n
    ans
  }
  if (is.null(seed)) draw() else with_local_seed(seed, draw())
}

match_lhs_to_candidates <- function(lhs, candidates) {
  selected <- integer(0)
  for (i in seq_len(nrow(lhs))) {
    d2 <- rowSums((sweep(candidates, 2, lhs[i, ], "-"))^2)
    if (length(selected) > 0L) d2[selected] <- Inf
    selected <- c(selected, which.min(d2))
  }
  selected
}

maximin_lhs_indices <- function(n, candidates, n_starts, seed) {
  with_local_seed(seed, {
    best_idx <- NULL
    best_score <- -Inf
    for (s in seq_len(n_starts)) {
      idx <- match_lhs_to_candidates(lhs_unit(n, ncol(candidates)), candidates)
      D <- as.matrix(stats::dist(candidates[idx, , drop = FALSE]))
      diag(D) <- Inf
      score <- min(D)
      if (score > best_score) {
        best_score <- score
        best_idx <- idx
      }
    }
    best_idx
  })
}

derive_eta <- function(candidates, initial_idx, cfg) {
  D <- as.matrix(stats::dist(candidates[initial_idx, , drop = FALSE]))
  diag(D) <- Inf
  r0 <- median(apply(D, 1, min))
  if (!is.finite(r0) || r0 <= 0) stop("Invalid nearest-neighbor radius.")
  eta0 <- -log(cfg$decay_at_r0) / r0^2
  list(r0 = r0, eta0 = eta0, eta = cfg$eta_multiplier * eta0)
}

format_mean_sd <- function(x, digits = 4L) {
  sprintf(
    paste0("%.", digits, "f (%.", digits, "f)"),
    mean(x, na.rm = TRUE), stats::sd(x, na.rm = TRUE)
  )
}

# ==============================================================================
# 3. Functional response and induced scalar objective
# ==============================================================================

make_basis_objects <- function(t, cfg) {
  B <- splines::bs(
    t, df = cfg$n_basis, degree = cfg$spline_degree,
    intercept = TRUE, Boundary.knots = range(t)
  )
  H <- ncol(B)
  D2 <- diff(diag(H), differences = 2)
  A <- solve(
    crossprod(B) + cfg$basis_penalty * crossprod(D2),
    t(B)
  )

  trap <- trapezoid_weights(t)
  loss_weights <- trap / sum(trap)
  domain_weights <- trap / sum(trap)

  loss_basis_cross <- crossprod(B, B * loss_weights)
  integrated_basis_cross <- crossprod(B, B * domain_weights)

  list(
    B = B,
    A = A,
    loss_weights = loss_weights,
    domain_weights = domain_weights,
    loss_basis_cross = loss_basis_cross,
    integrated_basis_cross = integrated_basis_cross,
    integrated_basis_sq = diag(integrated_basis_cross)
  )
}

true_mean_curve <- function(u, t) {
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
  mu_J <- rowSums((D %*% Q) * D) + as.vector(var_coef %*% diag(Q))

  var_J <- numeric(nrow(D))
  for (j in seq_len(nrow(D))) {
    v <- pmax(var_coef[j, ], 0)
    qd <- as.vector(Q %*% D[j, ])
    var_J[j] <- 2 * sum((Q^2) * tcrossprod(v, v)) +
      4 * sum(v * qd^2)
  }
  list(mu_J = mu_J, var_J = pmax(var_J, 1e-12))
}

objective_noise_variance <- function(mu_coef, sim) {
  Q <- sim$basis$loss_basis_cross
  Sigma <- sim$coefficient_measurement_noise_matrix
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

  candidate_unit <- lhs_unit(
    cfg$candidate_size - 1L, cfg$d, seed = cfg$seed + 101L
  )
  target_unit <- (cfg$target_u - cfg$input_lower) /
    (cfg$input_upper - cfg$input_lower)
  candidate_unit <- rbind(target_unit, candidate_unit)
  candidates <- sweep(
    sweep(candidate_unit, 2, cfg$input_upper - cfg$input_lower, "*"),
    2, cfg$input_lower, "+"
  )
  colnames(candidates) <- paste0("u", seq_len(cfg$d))
  colnames(candidate_unit) <- colnames(candidates)

  true_mean <- t(apply(candidates, 1, true_mean_curve, t = t))
  target_curve <- true_mean_curve(cfg$target_u, t)
  target_coef <- as.vector(basis$A %*% target_curve)
  true_coef <- true_mean %*% t(basis$A)

  sim_loss <- list(basis = basis, target_coef = target_coef)
  true_J <- objective_from_coefficients(true_coef, sim_loss)
  true_best_idx <- which.min(true_J)

  Sigma_measurement <- diag(cfg$noise_sd^2, cfg$n_time)
  coefficient_measurement_noise_matrix <-
    basis$A %*% Sigma_measurement %*% t(basis$A)
  coefficient_noise <- pmax(
    diag(coefficient_measurement_noise_matrix), 1e-10
  )

  ref_lhs <- lhs_unit(
    min(cfg$diversity_reference, cfg$candidate_size), cfg$d,
    seed = cfg$seed + 202L
  )
  reference_idx <- match_lhs_to_candidates(ref_lhs, candidate_unit)

  list(
    rep_id = rep_id,
    cfg = cfg,
    t = t,
    basis = basis,
    U = candidates,
    U_unit = candidate_unit,
    true_mean = true_mean,
    true_coef = true_coef,
    true_J = true_J,
    true_best_idx = true_best_idx,
    target_curve = target_curve,
    target_coef = target_coef,
    coefficient_noise = coefficient_noise,
    coefficient_measurement_noise_matrix =
      coefficient_measurement_noise_matrix,
    reference_idx = reference_idx
  )
}

simulate_observation <- function(sim, candidate_idx, evaluation_number) {
  seed <- sim$cfg$seed + sim$rep_id * 1000003L +
    candidate_idx * 1009L + evaluation_number * 97L
  with_local_seed(seed, {
    sim$true_mean[candidate_idx, ] +
      rnorm(sim$cfg$n_time, 0, sim$cfg$noise_sd)
  })
}

observe_indices <- function(sim, indices, initial_counts = NULL) {
  counts <- if (is.null(initial_counts)) {
    integer(nrow(sim$U))
  } else {
    initial_counts
  }
  Y <- matrix(NA_real_, nrow = length(indices), ncol = sim$cfg$n_time)
  for (i in seq_along(indices)) {
    idx <- indices[i]
    counts[idx] <- counts[idx] + 1L
    Y[i, ] <- simulate_observation(sim, idx, counts[idx])
  }
  list(Y = Y, counts = counts)
}

# ==============================================================================
# 4. GPFR surrogate with controlled mean-specification comparison
# ==============================================================================

trend_matrix <- function(U_unit, mean_specification) {
  U_unit <- as.matrix(U_unit)
  z <- 2 * U_unit - 1
  linear <- cbind(Intercept = 1, z)
  colnames(linear) <- c("Intercept", paste0("u", seq_len(ncol(z))))

  if (mean_specification == "Linear") return(linear)
  if (mean_specification != "Nonlinear") {
    stop("Unknown mean specification: ", mean_specification)
  }

  squares <- z^2
  colnames(squares) <- paste0("u", seq_len(ncol(z)), "_sq")
  pairs <- combn(ncol(z), 2)
  interactions <- vapply(seq_len(ncol(pairs)), function(k) {
    z[, pairs[1, k]] * z[, pairs[2, k]]
  }, numeric(nrow(z)))
  if (is.vector(interactions)) interactions <- matrix(interactions, ncol = 1L)
  colnames(interactions) <- apply(pairs, 2, function(pair) {
    paste0("u", pair[1], "_u", pair[2])
  })
  cbind(linear, squares, interactions)
}

mean_ridge_penalty <- function(mean_specification, q, cfg) {
  lambda <- if (mean_specification == "Nonlinear") {
    cfg$mean_ridge_nonlinear
  } else {
    cfg$mean_ridge_linear
  }
  penalty <- diag(lambda, q)
  penalty[1, 1] <- 0
  penalty
}

squared_distance <- function(A, B) {
  A <- as.matrix(A)
  B <- as.matrix(B)
  pmax(
    outer(rowSums(A^2), rowSums(B^2), "+") - 2 * tcrossprod(A, B),
    0
  )
}

kernel_from_inputs <- function(Ua, Ub, cfg) {
  d2 <- squared_distance(Ua / cfg$kernel_ell_u, Ub / cfg$kernel_ell_u)
  exp(-0.5 * d2)
}

fit_functional_gp <- function(Y, obs_idx, sim, mean_specification) {
  cfg <- sim$cfg
  C <- Y %*% t(sim$basis$A)
  Hm <- trend_matrix(
    sim$U_unit[obs_idx, , drop = FALSE], mean_specification
  )
  penalty <- mean_ridge_penalty(mean_specification, ncol(Hm), cfg)
  R <- kernel_from_inputs(
    sim$U_unit[obs_idx, , drop = FALSE],
    sim$U_unit[obs_idx, , drop = FALSE],
    cfg
  )
  fits <- vector("list", ncol(C))

  for (h in seq_len(ncol(C))) {
    y <- C[, h]
    beta_ridge <- solve(crossprod(Hm) + penalty, crossprod(Hm, y))
    residual_ridge <- y - as.vector(Hm %*% beta_ridge)
    noise <- sim$coefficient_noise[h]
    signal <- max(
      stats::var(residual_ridge) - noise,
      cfg$signal_floor_ratio * noise,
      1e-8
    )

    K <- signal * R + diag(noise, length(y))
    Uchol <- safe_chol(K, cfg$jitter)
    Kinv <- chol2inv(Uchol)
    XtKiX <- crossprod(Hm, Kinv %*% Hm)
    Cbeta <- solve(XtKiX + penalty)
    beta <- Cbeta %*% crossprod(Hm, Kinv %*% y)
    residual <- y - as.vector(Hm %*% beta)
    alpha <- Kinv %*% residual

    fits[[h]] <- list(
      signal = signal,
      noise = noise,
      beta = beta,
      alpha = alpha,
      Kinv = Kinv,
      H_train = Hm,
      Cbeta = Cbeta
    )
  }

  list(
    fits = fits,
    obs_idx = obs_idx,
    mean_specification = mean_specification,
    coefficient_observed = C,
    sim = sim
  )
}

predict_functional_gp <- function(model, pred_idx, return_curve = FALSE) {
  sim <- model$sim
  cfg <- sim$cfg
  Hs <- trend_matrix(
    sim$U_unit[pred_idx, , drop = FALSE], model$mean_specification
  )
  Rst <- kernel_from_inputs(
    sim$U_unit[pred_idx, , drop = FALSE],
    sim$U_unit[model$obs_idx, , drop = FALSE],
    cfg
  )

  Hcoef <- length(model$fits)
  mu_coef <- matrix(NA_real_, nrow = length(pred_idx), ncol = Hcoef)
  var_coef <- matrix(NA_real_, nrow = length(pred_idx), ncol = Hcoef)

  for (h in seq_len(Hcoef)) {
    fit <- model$fits[[h]]
    Kst <- fit$signal * Rst
    mu_coef[, h] <- as.vector(Hs %*% fit$beta + Kst %*% fit$alpha)

    base_reduction <- rowSums((Kst %*% fit$Kinv) * Kst)
    trend_residual <- Hs - Kst %*% fit$Kinv %*% fit$H_train
    trend_variance <- rowSums(
      (trend_residual %*% fit$Cbeta) * trend_residual
    )
    var_coef[, h] <- pmax(
      fit$signal - base_reduction + trend_variance,
      1e-12
    )
  }

  objective <- objective_moments(mu_coef, var_coef, sim)
  result <- list(
    mu_coef = mu_coef,
    var_coef = var_coef,
    mu_J = objective$mu_J,
    var_J = objective$var_J
  )

  if (return_curve) {
    result$mu_curve <- mu_coef %*% t(sim$basis$B)
    result$var_curve <- var_coef %*% t(sim$basis$B^2)
    result$sd_curve <- sqrt(pmax(result$var_curve, 0))
  }
  result
}

posterior_crosscov <- function(model, coefficient, idx_a, idx_b) {
  sim <- model$sim
  cfg <- sim$cfg
  fit <- model$fits[[coefficient]]

  R_ab <- kernel_from_inputs(
    sim$U_unit[idx_a, , drop = FALSE],
    sim$U_unit[idx_b, , drop = FALSE],
    cfg
  )
  R_at <- kernel_from_inputs(
    sim$U_unit[idx_a, , drop = FALSE],
    sim$U_unit[model$obs_idx, , drop = FALSE],
    cfg
  )
  R_bt <- kernel_from_inputs(
    sim$U_unit[idx_b, , drop = FALSE],
    sim$U_unit[model$obs_idx, , drop = FALSE],
    cfg
  )
  K_at <- fit$signal * R_at
  K_bt <- fit$signal * R_bt

  Ha <- trend_matrix(
    sim$U_unit[idx_a, , drop = FALSE], model$mean_specification
  )
  Hb <- trend_matrix(
    sim$U_unit[idx_b, , drop = FALSE], model$mean_specification
  )
  ra <- Ha - K_at %*% fit$Kinv %*% fit$H_train
  rb <- Hb - K_bt %*% fit$Kinv %*% fit$H_train

  fit$signal * R_ab -
    K_at %*% fit$Kinv %*% t(K_bt) +
    ra %*% fit$Cbeta %*% t(rb)
}

# ==============================================================================
# 5. Acquisition functions
# ==============================================================================

weight_decay <- function(pred_all, counts, sim) {
  cfg <- sim$cfg
  evaluated <- which(counts > 0L)
  objective_noise <- objective_noise_variance(pred_all$mu_coef, sim)
  tau_J <- cfg$kappa * objective_noise
  active <- pred_all$var_J[evaluated] <= tau_J[evaluated]

  if (!any(active)) {
    return(list(W = rep(1, nrow(sim$U)), active = active, tau_J = tau_J))
  }

  log_W <- numeric(nrow(sim$U))
  for (position in which(active)) {
    idx <- evaluated[position]
    d2 <- rowSums(
      (sweep(sim$U_unit, 2, sim$U_unit[idx, ], "-"))^2
    )
    local_weight <- 1 - (1 - cfg$rho) * exp(-cfg$eta * d2)
    log_W <- log_W + counts[idx] * log(pmax(local_weight, 1e-12))
  }
  list(W = exp(log_W), active = active, tau_J = tau_J)
}

global_profile_learning_gain <- function(model, pred_all, sim) {
  ref <- sim$reference_idx
  all_idx <- seq_len(nrow(sim$U))
  H <- length(model$fits)
  J <- length(all_idx)

  cross_array <- array(NA_real_, dim = c(length(ref), J, H))
  for (h in seq_len(H)) {
    cross_array[, , h] <- posterior_crosscov(model, h, ref, all_idx)
  }

  Q_B <- sim$basis$integrated_basis_cross
  gain <- numeric(J)
  for (j in seq_len(J)) {
    future_covariance <- diag(pred_all$var_coef[j, ], H) +
      sim$coefficient_measurement_noise_matrix
    future_chol <- safe_chol(future_covariance, sim$cfg$jitter)
    future_inverse <- chol2inv(future_chol)
    cross_j <- matrix(cross_array[, j, ], nrow = length(ref), ncol = H)
    cross_gram <- crossprod(cross_j) / length(ref)
    gain[j] <- sum(Q_B * future_inverse * cross_gram)
  }
  pmax(gain, 0)
}

expected_quantile_improvement <- function(
    pred_all, evaluated_idx, objective_noise, beta) {
  z_beta <- qnorm(beta)
  current_quantile <- pred_all$mu_J +
    z_beta * sqrt(pmax(pred_all$var_J, 0))
  incumbent <- min(current_quantile[evaluated_idx])

  variance <- pmax(pred_all$var_J, 1e-12)
  noise <- pmax(objective_noise, 1e-12)
  variance_after <- pmax(
    variance - variance^2 / (variance + noise),
    0
  )
  update_sd <- variance / sqrt(variance + noise)
  mean_after <- pred_all$mu_J + z_beta * sqrt(variance_after)
  delta <- incumbent - mean_after

  result <- numeric(length(delta))
  regular <- update_sd > 1e-12
  z <- delta[regular] / update_sd[regular]
  result[regular] <- delta[regular] * pnorm(z) +
    update_sd[regular] * dnorm(z)
  result[!regular] <- pmax(delta[!regular], 0)
  pmax(result, 0)
}

functional_acquisition <- function(method, model, counts, sim, iteration) {
  all_idx <- seq_len(nrow(sim$U))
  pred <- predict_functional_gp(model, all_idx)
  evaluated <- which(counts > 0L)
  incumbent <- min(pred$mu_J[evaluated])

  if (method == "WDEI") {
    draws <- sample_objective_draws(
      pred, sim, sim$cfg$objective_mc,
      seed = sim$cfg$seed + sim$rep_id * 104729L + iteration * 1009L
    )
    FEI <- rowMeans(pmax(incumbent - draws, 0))
    decay <- weight_decay(pred, counts, sim)
    D <- global_profile_learning_gain(model, pred, sim)
    weighted_FEI <- decay$W * FEI
    c_I <- max(weighted_FEI) + sim$cfg$utility_epsilon
    c_D <- max(D) + sim$cfg$utility_epsilon
    score <- (1 - sim$cfg$alpha) * weighted_FEI / c_I +
      sim$cfg$alpha * D / c_D
  } else if (method == "GPFR-EQI") {
    FEI <- rep(NA_real_, nrow(sim$U))
    D <- rep(NA_real_, nrow(sim$U))
    decay <- weight_decay(pred, counts, sim)
    objective_noise <- objective_noise_variance(pred$mu_coef, sim)
    score <- expected_quantile_improvement(
      pred, evaluated, objective_noise, sim$cfg$eqi_beta
    )
  } else {
    stop("Unknown sequential method: ", method)
  }

  list(
    score = score,
    FEI = FEI,
    W = decay$W,
    D = D,
    pred = pred,
    tau_J = decay$tau_J,
    active_decay = sum(decay$active)
  )
}

# ==============================================================================
# 6. Sequential evaluation and performance measures
# ==============================================================================

select_recommendation <- function(model, evaluated_idx, sim,
                                  return_curve = FALSE) {
  pred <- predict_functional_gp(model, evaluated_idx, return_curve)
  local <- which.min(pred$mu_J)
  list(local = local, index = evaluated_idx[local], pred = pred)
}

recommendation_pog <- function(model, counts, sim) {
  evaluated <- which(counts > 0L)
  rec <- select_recommendation(model, evaluated, sim)
  sim$true_J[rec$index] - sim$true_J[sim$true_best_idx]
}

integrated_latent_variance <- function(pred, sim) {
  weights <- matrix(
    sim$basis$domain_weights,
    nrow = nrow(pred$var_curve),
    ncol = sim$cfg$n_time,
    byrow = TRUE
  )
  mean(rowSums(pred$var_curve * weights))
}

run_sequential_method <- function(method, mean_specification, sim,
                                  initial_idx, initial_Y) {
  cfg <- sim$cfg
  obs_idx <- initial_idx
  Y <- initial_Y
  counts <- tabulate(obs_idx, nbins = nrow(sim$U))
  n_add <- cfg$N_total - cfg$M_init
  model <- fit_functional_gp(Y, obs_idx, sim, mean_specification)

  history <- data.frame(
    Rep = sim$rep_id,
    Mean_Specification = mean_specification,
    Method = method,
    Iteration = 0:n_add,
    POG = NA_real_,
    Selected = NA_integer_,
    Is_Repeat = 0L,
    Useful_Repeat = 0L,
    Redundant_Repeat = 0L,
    Active_Decay_Locations = NA_integer_,
    stringsAsFactors = FALSE
  )
  history$POG[1] <- recommendation_pog(model, counts, sim)

  for (iteration in seq_len(n_add)) {
    acquisition <- functional_acquisition(
      method, model, counts, sim, iteration
    )
    if (!any(is.finite(acquisition$score))) {
      stop(
        "All acquisition values are invalid for ", method,
        ", replication ", sim$rep_id,
        ", iteration ", iteration, "."
      )
    }

    selected <- which.max(acquisition$score)
    is_repeat <- as.integer(counts[selected] > 0L)
    selected_variance <- acquisition$pred$var_J[selected]
    selected_threshold <- acquisition$tau_J[selected]
    useful_repeat <- as.integer(
      is_repeat == 1L && selected_variance > selected_threshold
    )
    redundant_repeat <- as.integer(
      is_repeat == 1L && selected_variance <= selected_threshold
    )

    counts[selected] <- counts[selected] + 1L
    Y <- rbind(
      Y,
      simulate_observation(sim, selected, counts[selected])
    )
    obs_idx <- c(obs_idx, selected)
    model <- fit_functional_gp(Y, obs_idx, sim, mean_specification)

    row <- iteration + 1L
    history$POG[row] <- recommendation_pog(model, counts, sim)
    history$Selected[row] <- selected
    history$Is_Repeat[row] <- is_repeat
    history$Useful_Repeat[row] <- useful_repeat
    history$Redundant_Repeat[row] <- redundant_repeat
    history$Active_Decay_Locations[row] <- acquisition$active_decay

    if (cfg$verbose && iteration %% 10L == 0L) {
      cat(sprintf(
        "  Rep %02d | %-9s | %-8s | iteration %02d/%02d\n",
        sim$rep_id, mean_specification, method, iteration, n_add
      ))
    }
  }

  list(
    method = method,
    mean_specification = mean_specification,
    obs_idx = obs_idx,
    Y = Y,
    counts = counts,
    model = model,
    history = history
  )
}

evaluate_method <- function(method, mean_specification, model,
                            obs_idx, counts, sim, history = NULL) {
  all_idx <- seq_len(nrow(sim$U))
  pred_all <- predict_functional_gp(model, all_idx, return_curve = TRUE)

  curve_error <- pred_all$mu_curve - sim$true_mean
  global_rmse <- mean(sqrt(rowMeans(curve_error^2)))
  integrated_variance <- integrated_latent_variance(pred_all, sim)

  evaluated <- which(counts > 0L)
  rec <- select_recommendation(
    model, evaluated, sim, return_curve = TRUE
  )
  rec_idx <- rec$index
  rec_local <- rec$local
  pog <- sim$true_J[rec_idx] - sim$true_J[sim$true_best_idx]
  optimal_rmse <- sqrt(mean(
    (rec$pred$mu_curve[rec_local, ] - sim$true_mean[rec_idx, ])^2
  ))

  data.frame(
    Rep = sim$rep_id,
    Mean_Specification = mean_specification,
    Method = method,
    Global_RMSE = global_rmse,
    Integrated_Latent_Variance = integrated_variance,
    POG = pog,
    Optimal_RMSE = optimal_rmse,
    Unique_Inputs = length(evaluated),
    Replicates = length(obs_idx) - length(evaluated),
    Useful_Repeats = if (is.null(history)) NA_real_ else
      sum(history$Useful_Repeat, na.rm = TRUE),
    Redundant_Repeats = if (is.null(history)) NA_real_ else
      sum(history$Redundant_Repeat, na.rm = TRUE),
    Recommended_Index = rec_idx,
    Recommended_Input = paste(
      sprintf("%.6f", sim$U[rec_idx, ]), collapse = ","
    ),
    True_Best_Index = sim$true_best_idx,
    True_Best_Input = paste(
      sprintf("%.6f", sim$U[sim$true_best_idx, ]), collapse = ","
    ),
    Eta = sim$cfg$eta,
    Initial_NN_Radius = sim$eta_scale$r0,
    stringsAsFactors = FALSE
  )
}

# ==============================================================================
# 7. Paired replication design
# ==============================================================================

run_replication <- function(rep_id, cfg = CFG) {
  sim <- make_simulation(rep_id, cfg)
  initial_idx <- maximin_lhs_indices(
    cfg$M_init, sim$U_unit, n_starts = 200L,
    seed = cfg$seed + rep_id * 7919L
  )
  eta_scale <- derive_eta(sim$U_unit, initial_idx, cfg)
  sim$cfg$eta <- eta_scale$eta
  sim$eta_scale <- eta_scale

  initial_observations <- observe_indices(sim, initial_idx)
  initial_Y <- initial_observations$Y

  metrics <- list()
  histories <- list()
  metric_counter <- 1L
  history_counter <- 1L

  nonlinear_initial_model <- fit_functional_gp(
    initial_Y, initial_idx, sim, "Nonlinear"
  )
  nonlinear_initial_counts <- tabulate(
    initial_idx, nbins = nrow(sim$U)
  )
  metrics[[metric_counter]] <- evaluate_method(
    "Initial LHS", "Nonlinear", nonlinear_initial_model,
    initial_idx, nonlinear_initial_counts, sim
  )
  metric_counter <- metric_counter + 1L

  one_shot_idx <- maximin_lhs_indices(
    cfg$N_total, sim$U_unit, n_starts = 200L,
    seed = cfg$seed + rep_id * 3571L
  )
  one_shot_observations <- observe_indices(sim, one_shot_idx)
  one_shot_model <- fit_functional_gp(
    one_shot_observations$Y, one_shot_idx, sim, "Nonlinear"
  )
  metrics[[metric_counter]] <- evaluate_method(
    "One-shot LHS", "Nonlinear", one_shot_model,
    one_shot_idx, one_shot_observations$counts, sim
  )
  metric_counter <- metric_counter + 1L

  for (mean_specification in MEAN_SPECIFICATIONS) {
    for (method in SEQUENTIAL_METHODS) {
      fit <- run_sequential_method(
        method, mean_specification, sim, initial_idx, initial_Y
      )
      metrics[[metric_counter]] <- evaluate_method(
        method, mean_specification, fit$model,
        fit$obs_idx, fit$counts, sim, fit$history
      )
      histories[[history_counter]] <- fit$history
      metric_counter <- metric_counter + 1L
      history_counter <- history_counter + 1L
    }
  }

  list(
    metrics = do.call(rbind, metrics),
    history = do.call(rbind, histories)
  )
}

run_experiment <- function(n_rep = N_REP, cfg = CFG) {
  all_metrics <- vector("list", n_rep)
  all_histories <- vector("list", n_rep)
  start_time <- proc.time()[3]

  for (rep_id in seq_len(n_rep)) {
    cat(sprintf("\nAppendix C replication %d/%d\n", rep_id, n_rep))
    result <- run_replication(rep_id, cfg)
    all_metrics[[rep_id]] <- result$metrics
    all_histories[[rep_id]] <- result$history
  }

  list(
    raw = do.call(rbind, all_metrics),
    history = do.call(rbind, all_histories),
    elapsed_seconds = proc.time()[3] - start_time
  )
}

# ==============================================================================
# 8. Table C1 and POG convergence figure
# ==============================================================================

result_row_order <- data.frame(
  Mean_Specification = c(
    rep("Nonlinear", 4),
    rep("Linear", 2)
  ),
  Method = c(
    "Initial LHS", "One-shot LHS", "GPFR-EQI", "WDEI",
    "GPFR-EQI", "WDEI"
  ),
  stringsAsFactors = FALSE
)

build_numeric_summary <- function(raw) {
  metrics <- c(
    "Global_RMSE",
    "Integrated_Latent_Variance",
    "POG",
    "Optimal_RMSE",
    "Replicates",
    "Useful_Repeats",
    "Redundant_Repeats"
  )
  rows <- list()
  counter <- 1L

  for (i in seq_len(nrow(result_row_order))) {
    specification <- result_row_order$Mean_Specification[i]
    method <- result_row_order$Method[i]
    sub <- raw[
      raw$Mean_Specification == specification & raw$Method == method,
      , drop = FALSE
    ]
    if (nrow(sub) == 0L) next

    row <- data.frame(
      Mean_Specification = specification,
      Method = method,
      N_Replications = nrow(sub),
      stringsAsFactors = FALSE
    )
    for (metric in metrics) {
      values <- sub[[metric]]
      row[[paste0(metric, "_Mean")]] <- if (all(is.na(values))) {
        NA_real_
      } else {
        mean(values, na.rm = TRUE)
      }
      row[[paste0(metric, "_SD")]] <- if (all(is.na(values))) {
        NA_real_
      } else {
        stats::sd(values, na.rm = TRUE)
      }
    }
    rows[[counter]] <- row
    counter <- counter + 1L
  }
  do.call(rbind, rows)
}

build_table_c1 <- function(raw) {
  rows <- list()
  counter <- 1L

  for (i in seq_len(nrow(result_row_order))) {
    specification <- result_row_order$Mean_Specification[i]
    method <- result_row_order$Method[i]
    sub <- raw[
      raw$Mean_Specification == specification & raw$Method == method,
      , drop = FALSE
    ]
    if (nrow(sub) == 0L) next

    replicate_cell <- if (method %in% SEQUENTIAL_METHODS) {
      sprintf(
        "%.2f (%.2f/%.2f)",
        mean(sub$Replicates),
        mean(sub$Useful_Repeats),
        mean(sub$Redundant_Repeats)
      )
    } else {
      "--"
    }

    rows[[counter]] <- data.frame(
      `Mean specification` = if (specification == "Nonlinear") {
        "Nonlinear (formal)"
      } else {
        "Linear (diagnostic)"
      },
      Method = method,
      `Global RMSE` = format_mean_sd(sub$Global_RMSE),
      `Integrated latent variance` =
        format_mean_sd(sub$Integrated_Latent_Variance),
      POG = format_mean_sd(sub$POG),
      `Optimal RMSE` = format_mean_sd(sub$Optimal_RMSE),
      `Replicates (useful/redundant)` = replicate_cell,
      check.names = FALSE,
      stringsAsFactors = FALSE
    )
    counter <- counter + 1L
  }
  do.call(rbind, rows)
}

summarize_pog_history <- function(history) {
  keys <- unique(history[, c(
    "Mean_Specification", "Method", "Iteration"
  )])
  rows <- vector("list", nrow(keys))

  for (i in seq_len(nrow(keys))) {
    values <- history$POG[
      history$Mean_Specification == keys$Mean_Specification[i] &
        history$Method == keys$Method[i] &
        history$Iteration == keys$Iteration[i]
    ]
    n <- sum(is.finite(values))
    mean_value <- mean(values, na.rm = TRUE)
    se <- if (n > 1L) stats::sd(values, na.rm = TRUE) / sqrt(n) else 0
    rows[[i]] <- data.frame(
      Mean_Specification = keys$Mean_Specification[i],
      Method = keys$Method[i],
      Iteration = keys$Iteration[i],
      Mean = mean_value,
      SE = se,
      Lower = pmax(mean_value - 1.96 * se, 0),
      Upper = mean_value + 1.96 * se,
      N = n,
      stringsAsFactors = FALSE
    )
  }
  do.call(rbind, rows)
}

plot_pog_convergence <- function(history_summary) {
  oldpar <- par(no.readonly = TRUE)
  on.exit(par(oldpar), add = TRUE)
  set_plot_style(mfrow = c(1, 2), mar = c(4.4, 4.8, 2.4, 0.7))

  colors <- c("GPFR-EQI" = "#D55E00", "WDEI" = "#0072B2")
  specifications <- c("Linear", "Nonlinear")
  panel_titles <- c(
    "Linear mean (diagnostic)",
    "Nonlinear mean (formal)"
  )

  for (panel in seq_along(specifications)) {
    specification <- specifications[panel]
    panel_data <- history_summary[
      history_summary$Mean_Specification == specification,
      , drop = FALSE
    ]
    ylim <- range(c(0, panel_data$Upper), finite = TRUE)
    if (diff(ylim) <= 0) ylim <- c(0, 1)

    plot(
      range(panel_data$Iteration), ylim,
      type = "n",
      xlab = "Sequential iteration",
      ylab = "Recommendation POG",
      main = panel_titles[panel]
    )
    grid(col = "grey88")

    for (method in SEQUENTIAL_METHODS) {
      sub <- panel_data[panel_data$Method == method, , drop = FALSE]
      sub <- sub[order(sub$Iteration), ]
      polygon(
        c(sub$Iteration, rev(sub$Iteration)),
        c(sub$Lower, rev(sub$Upper)),
        col = grDevices::adjustcolor(colors[method], alpha.f = 0.16),
        border = NA
      )
      lines(
        sub$Iteration, sub$Mean,
        col = colors[method], lwd = 2.2
      )
    }

    legend(
      "topright",
      legend = SEQUENTIAL_METHODS,
      col = colors[SEQUENTIAL_METHODS],
      lwd = 2.2,
      bty = "n",
      cex = 0.90
    )
    mtext(
      paste0("(", letters[panel], ")"),
      side = 3, adj = 0, line = 0.25, font = 2
    )
  }
  invisible(NULL)
}

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

save_pog_figure <- function(history_summary, output_dir) {
  pdf_path <- file.path(output_dir, "FigureC1_POG_Convergence.pdf")
  tiff_path <- file.path(output_dir, "FigureC1_POG_Convergence.tiff")

  open_pdf_device(pdf_path, width = 8.2, height = 3.8)
  plot_pog_convergence(history_summary)
  invisible(dev.off())

  grDevices::tiff(
    filename = tiff_path,
    width = 8.2,
    height = 3.8,
    units = "in",
    res = 600,
    compression = "lzw"
  )
  plot_pog_convergence(history_summary)
  invisible(dev.off())
}

# ==============================================================================
# 9. Formal execution and export
# ==============================================================================

RESULTS <- run_experiment(N_REP, CFG)
TABLE_C1 <- build_table_c1(RESULTS$raw)
TABLE_C1_NUMERIC <- build_numeric_summary(RESULTS$raw)
POG_HISTORY_SUMMARY <- summarize_pog_history(RESULTS$history)

cat("\n================ Table C1: six-dimensional example ================\n")
print(TABLE_C1, row.names = FALSE, right = FALSE)
cat(sprintf(
  "\nCompleted %d paired replications in %.2f minutes.\n",
  N_REP, RESULTS$elapsed_seconds / 60
))

if (SAVE_OUTPUTS) {
  if (!dir.exists(OUTPUT_DIR)) dir.create(OUTPUT_DIR, recursive = TRUE)

  write.csv(
    TABLE_C1,
    file.path(OUTPUT_DIR, "TableC1_Appendix6D.csv"),
    row.names = FALSE,
    fileEncoding = "UTF-8"
  )
  write.csv(
    TABLE_C1_NUMERIC,
    file.path(OUTPUT_DIR, "TableC1_Appendix6D_Numeric.csv"),
    row.names = FALSE,
    fileEncoding = "UTF-8"
  )
  write.csv(
    RESULTS$raw,
    file.path(OUTPUT_DIR, "AppendixC_Raw.csv"),
    row.names = FALSE,
    fileEncoding = "UTF-8"
  )
  write.csv(
    RESULTS$history,
    file.path(OUTPUT_DIR, "AppendixC_POG_History.csv"),
    row.names = FALSE,
    fileEncoding = "UTF-8"
  )
  write.csv(
    POG_HISTORY_SUMMARY,
    file.path(OUTPUT_DIR, "AppendixC_POG_History_Summary.csv"),
    row.names = FALSE,
    fileEncoding = "UTF-8"
  )
  save_pog_figure(POG_HISTORY_SUMMARY, OUTPUT_DIR)
}
