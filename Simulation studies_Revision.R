# ==============================================================================
# 0. 
# ==============================================================================
packages <- c("GPFDA", "MASS", "lhs", "doParallel", "foreach", "GA", "DiceKriging")
new_packages <- packages[!(packages %in% installed.packages()[,"Package"])]
if(length(new_packages)) install.packages(new_packages)

library(GPFDA)
library(MASS)
library(lhs)
library(doParallel)
library(foreach)
library(GA)
library(DiceKriging) 

rm(list = ls())
set.seed(20260212) 

# ==============================================================================
# 1. N=40
# ==============================================================================
NOISE_SD <- 0.5 
n <- 41           
M_init <- 20
N_total <- 40     
N_add <- N_total - M_init
MIN_DIST_TOL <- 1e-4

#
w_div <- 0.5     
w_ei <- 1 - w_div

hp <- list('pow.ex.v' = log(10), 'pow.ex.w' = log(1), 'vv' = log(1))
tt <- seq(-2, 2, len = n)
#
a <- cos((tt)^2)
b <- sin((0.5 * tt)^3)

param_ranges <- list(u0 = c(-1, 1), u1 = c(6, 8))

# ------------------------------------------------------------------------------
#
# ------------------------------------------------------------------------------
generate_data <- function(u0, u1, n_points, type = "train") {
  x_base <- exp(tt)
  if(type == "train") {
    x <- x_base + rnorm(n_points, 0, 0.001) 
  } else {
    x <- x_base
  }
  y_true <- u0 * a + u1 * b
  if(type == "train") {
    x_mat <- as.matrix(x)
    Sigma <- cov.pow.ex(hyper = hp, input = x_mat, gamma = 1)
    diag(Sigma) <- diag(Sigma) + exp(hp$vv)
    y_obs <- y_true + mvrnorm(n = 1, mu = rep(0, n_points), Sigma = Sigma) + rnorm(n_points, 0, NOISE_SD)
  } else {
    y_obs <- y_true 
  }
  return(list(x = as.vector(x), y_obs = as.vector(y_obs), y_true = as.vector(y_true)))
}

# ------------------------------------------------------------------------------
# Envelope
# ------------------------------------------------------------------------------
grid_u0 <- seq(-1, 1, length.out=101)
grid_u1 <- seq(6, 8, length.out=101)
grid_all <- expand.grid(u0=grid_u0, u1=grid_u1)
y_all_grid <- matrix(NA, nrow(grid_all), n)
for(i in 1:nrow(grid_all)) {
  y_all_grid[i,] <- grid_all[i,1]*a + grid_all[i,2]*b
}
true_min_curve <- apply(y_all_grid, 2, min)

# ==============================================================================
# 2. 
# ==============================================================================
u0_test <- seq(-1, 1, length.out = 25)
u1_test <- seq(6, 8, length.out = 25)
test_grid <- expand.grid(u0 = u0_test, u1 = u1_test)
M_pre <- nrow(test_grid)
scalar_test <- as.matrix(test_grid)
colnames(scalar_test) <- c("u0", "u1")
# 
x_test_standard <- matrix(NA, M_pre, n)
y_test_true <- matrix(NA, M_pre, n) 

for (i in 1:M_pre) {
  d <- generate_data(scalar_test[i,1], scalar_test[i,2], n, type = "test")
  x_test_standard[i,] <- d$x
  y_test_true[i,] <- d$y_true
}

# ==============================================================================
# 3. Method 1
# ==============================================================================
cat("Step 1: Initial Sampling...\n")
lhs_sample <- improvedLHS(M_init, 2)
u_init <- cbind(
  lhs_sample[, 1] * diff(param_ranges$u0) + param_ranges$u0[1],
  lhs_sample[, 2] * diff(param_ranges$u1) + param_ranges$u1[1]
)
colnames(u_init) <- c("u0", "u1")
initial_points_plot <- u_init 

x_init <- matrix(0, M_init, n); y_init <- matrix(0, M_init, n)
y_init_true <- matrix(0, M_init, n) 

for(i in 1:M_init) {
  d <- generate_data(u_init[i,1], u_init[i,2], n, type = "train")
  x_init[i,] <- d$x; y_init[i,] <- d$y_obs; y_init_true[i,] <- d$y_true
}

uCoefList <- list(list(lambda=1e-4, nbasi=10), list(lambda=1e-4, nbasi=10))
fyList_set <- list(nbasis=10, lambda=1e-4)
model_init <- gpfr(response=y_init, time=tt, uReg=u_init, gpReg=x_init, fyList=fyList_set, uCoefList=uCoefList, Cov='pow.ex', gamma=1, fitting=TRUE)

# ==============================================================================
# 4. 
# ==============================================================================

# --- Method 2: One-Shot LHS ---
cat("Step 2: One-Shot LHS...\n")
lhs_os <- improvedLHS(N_total, 2)
u_os <- cbind(lhs_os[,1]*diff(param_ranges$u0)+param_ranges$u0[1], lhs_os[,2]*diff(param_ranges$u1)+param_ranges$u1[1])
x_os <- matrix(0, N_total, n); y_os <- matrix(0, N_total, n); y_os_true <- matrix(0, N_total, n)
for(i in 1:N_total) {
  d <- generate_data(u_os[i,1], u_os[i,2], n, type = "train")
  x_os[i,] <- d$x; y_os[i,] <- d$y_obs; y_os_true[i,] <- d$y_true
}
model_os <- gpfr(response=y_os, time=tt, uReg=u_os, gpReg=x_os, fyList=fyList_set, uCoefList=uCoefList, Cov='pow.ex', gamma=1, fitting=TRUE)

# --- Method 3: ParEGO (Safe Mode) ---
cat("Step 3: ParEGO (Safe Mode)...\n")
u_parego <- u_init; y_parego <- y_init; y_parego_true <- y_init_true 
parego_history <- list(max_EI = numeric(N_add))

for(iter in 1:N_add) {
  y_norm <- scale(y_parego, center=apply(y_parego, 2, min), scale=apply(y_parego, 2, max)-apply(y_parego, 2, min)+1e-6)
  w <- runif(n); w <- w/sum(w)
  f_scalar <- numeric(nrow(y_norm))
  for(k in 1:nrow(y_norm)) f_scalar[k] <- max(w * y_norm[k,]) + 0.05 * sum(w * y_norm[k,])
  
  model_k <- km(design=u_parego, response=f_scalar, covtype="matern5_2", control=list(trace=FALSE), nugget=1e-8)
  
  pred <- predict(model_k, newdata=data.frame(scalar_test), type="UK")
  d <- min(f_scalar) - pred$mean
  ei <- d*pnorm(d/pred$sd) + pred$sd*dnorm(d/pred$sd)
  
  parego_history$max_EI[iter] <- max(ei, na.rm=TRUE)
  
  #
  sorted_idx <- order(ei, decreasing = TRUE)
  best_idx <- -1
  
  for(idx in sorted_idx) {
    candidate <- scalar_test[idx,]
    candidate_mat <- matrix(candidate, nrow = nrow(u_parego), ncol = ncol(u_parego), byrow = TRUE)
    min_dist <- min(sqrt(rowSums((u_parego - candidate_mat)^2)))
    
    if(min_dist > MIN_DIST_TOL) {
      best_idx <- idx
      break
    }
  }
  if(best_idx == -1) best_idx <- sample(1:M_pre, 1) 
  
  new_u <- scalar_test[best_idx,]
  d_new <- generate_data(new_u[1], new_u[2], n, type = "train")
  u_parego <- rbind(u_parego, new_u); y_parego <- rbind(y_parego, d_new$y_obs); y_parego_true <- rbind(y_parego_true, d_new$y_true)
}
x_parego <- matrix(0, nrow(u_parego), n)
for(k in 1:nrow(u_parego)) x_parego[k,] <- generate_data(u_parego[k,1], u_parego[k,2], n, type = "train")$x
model_parego <- gpfr(response=y_parego, time=tt, uReg=u_parego, gpReg=x_parego, fyList=fyList_set, uCoefList=uCoefList, Cov='pow.ex', gamma=1, fitting=TRUE)

# --- Method 4: WDEI (Proposed - Rank Based) ---
cat("Step 4: WDEI (Rank-Based Scheme)...\n")
u_wdei <- u_init; y_wdei <- y_init; x_wdei <- x_init; y_wdei_true <- y_init_true 
visited <- c()

wdei_history <- list(
  added_points = matrix(NA, nrow=N_add, ncol=2), 
  max_S = numeric(N_add),        
  max_WDEI = numeric(N_add)  
)

for(iter in 1:N_add) {
  model_w <- gpfr(response=y_wdei, time=tt, uReg=u_wdei, gpReg=x_wdei, 
                  fyList=fyList_set, uCoefList=uCoefList, 
                  Cov='pow.ex', gamma=1, fitting=TRUE)
  
  mu_tr <- matrix(NA, nrow(u_wdei), n)
  for(k in 1:nrow(u_wdei)) {
    p_obj <- gpfrPredict(model_w, testInputGP=x_wdei[k,], testTime=tt, uReg=u_wdei[k,])
    mu_tr[k,] <- p_obj$ypred.mean
  }
  fmin_p <- apply(mu_tr, 2, min)
  
  mu_t <- matrix(NA, M_pre, n); sd_t <- matrix(NA, M_pre, n)
  for(i in 1:M_pre) {
    p <- gpfrPredict(model_w, testInputGP=x_test_standard[i,], testTime=tt, uReg=scalar_test[i,])
    mu_t[i,] <- p$ypred.mean; sd_t[i,] <- p$ypred.sd
  }
  
  d_mat <- t(fmin_p - t(mu_t)); dn_mat <- d_mat / sd_t
  ei_mat <- d_mat * pnorm(dn_mat) + sd_t * dnorm(dn_mat)
  ei_int <- rowSums(ei_mat) 
  
  # --- Rank  ---
  # 1. distance
  dists <- numeric(M_pre)
  for(i in 1:M_pre) if(i %in% visited) dists[i] <- 0 else dists[i] <- min(sqrt(rowSums((u_wdei - scalar_test[i,])^2)))
  
  # 2. Distance * Uncertainty
  avg_sd <- rowMeans(sd_t)
  diversity_val <- dists * avg_sd 
  
  # 3.
  rank_ei <- rank(ei_int, ties.method = "random") 
  rank_div <- rank(diversity_val, ties.method = "random")
  
  # 4. Rank  [0, 1]
  r_ei_n <- rank_ei / M_pre
  r_div_n <- rank_div / M_pre
  
  # 5. weight
  score <- w_ei * r_ei_n + w_div * r_div_n
  score[visited] <- -1
  
  # 6. 
  sorted_idx <- order(score, decreasing = TRUE)
  best_idx <- -1
  
  for(idx in sorted_idx) {
    candidate <- scalar_test[idx,]
    candidate_mat <- matrix(candidate, nrow = nrow(u_wdei), ncol = ncol(u_wdei), byrow = TRUE)
    min_dist <- min(sqrt(rowSums((u_wdei - candidate_mat)^2)))
    
    if(min_dist > MIN_DIST_TOL) {
      best_idx <- idx
      break
    }
  }
  if(best_idx == -1) best_idx <- sample(1:M_pre, 1) 
  
  visited <- c(visited, best_idx)
  new_u <- scalar_test[best_idx,]
  
  wdei_history$added_points[iter, ] <- new_u
  wdei_history$max_S[iter] <- max(score, na.rm=TRUE)
  wdei_history$max_WDEI[iter] <- max(ei_int, na.rm=TRUE) 
  
  d_new <- generate_data(new_u[1], new_u[2], n, type = "train")
  u_wdei <- rbind(u_wdei, new_u); x_wdei <- rbind(x_wdei, d_new$x); y_wdei <- rbind(y_wdei, d_new$y_obs)
  y_wdei_true <- rbind(y_wdei_true, d_new$y_true)
  
  if(iter %% 5 == 0) cat(iter, " ")
}
cat("\n")
model_wdei <- gpfr(response=y_wdei, time=tt, uReg=u_wdei, gpReg=x_wdei, fyList=fyList_set, uCoefList=uCoefList, Cov='pow.ex', gamma=1, fitting=TRUE)

# ==============================================================================
# 5. Results
# ==============================================================================
cat("Evaluating Results...\n")

get_metrics <- function(model, name) {
  p_means <- matrix(NA, M_pre, n)
  for(i in 1:M_pre) p_means[i,] <- gpfrPredict(model, testInputGP=x_test_standard[i,], testTime=tt, uReg=scalar_test[i,])$ypred.mean
  
  rmse_t2 <- sqrt(mean((p_means - y_test_true)^2))
  r2_t2 <- 1 - sum((y_test_true - p_means)^2) / sum((y_test_true - mean(y_test_true))^2)
  
  best_idx <- which.min(rowSums(p_means))
  best_pred_curve <- p_means[best_idx, ]; best_true_curve <- y_test_true[best_idx, ] 
  
  pog <- sqrt(mean((best_pred_curve - true_min_curve)^2))
  opt_rmse <- sqrt(mean((best_pred_curve - best_true_curve)^2))
  
  best_sd <- gpfrPredict(model, testInputGP=x_test_standard[best_idx,], testTime=tt, uReg=scalar_test[best_idx,])$ypred.sd
  
  # CP Calculation
  sim_obs <- best_true_curve + rnorm(n, 0, NOISE_SD)
  lower <- best_pred_curve - 1.96 * best_sd
  upper <- best_pred_curve + 1.96 * best_sd
  cp <- mean(sim_obs >= lower & sim_obs <= upper)
  
  return(list(Pred = c(rmse_t2, r2_t2), Opt = c(pog, opt_rmse, cp), Plot = list(Pred=best_pred_curve, True=best_true_curve, Min=true_min_curve, SD=best_sd, POG=pog, RMSE=opt_rmse)))
}

m1 <- get_metrics(model_init, "M1")
m2 <- get_metrics(model_os, "M2")
m3 <- get_metrics(model_parego, "M3")
m4 <- get_metrics(model_wdei, "M4")

# ==============================================================================
# 6. Table 1 & 2
# ==============================================================================
df1 <- data.frame(Method = c("One-Shot(M=10)", "One-Shot(N=30)", "ParEGO", "WDEI"),
                  Type2_RMSE = c(m1$Pred[1], m2$Pred[1], m3$Pred[1], m4$Pred[1]),
                  Type2_R2   = c(m1$Pred[2], m2$Pred[2], m3$Pred[2], m4$Pred[2]))

df2 <- data.frame(Method = c("One-Shot(M=10)", "One-Shot(N=30)", "ParEGO", "WDEI"),
                  POG = c(m1$Opt[1], m2$Opt[1], m3$Opt[1], m4$Opt[1]),
                  Opt_RMSE = c(m1$Opt[2], m2$Opt[2], m3$Opt[2], m4$Opt[2]),
                  Opt_CP = c(m1$Opt[3], m2$Opt[3], m3$Opt[3], m4$Opt[3]))

num_cols1 <- sapply(df1, is.numeric); df1[num_cols1] <- round(df1[num_cols1], 4)
num_cols2 <- sapply(df2, is.numeric); df2[num_cols2] <- round(df2[num_cols2], 4)

cat("\nTable 1: Generalization Performance (Type II)\n"); print(df1)
cat("\nTable 2: Optimization Performance\n"); print(df2)

# ==============================================================================
# 7. Figure 4 & 5
# ==============================================================================
par(family = "serif")
iter_seq <- 1:N_add

# Figure 4-1 (ParEGO)
par(mfrow=c(1,1), mar=c(5,5,4,2))
plot(iter_seq, parego_history$max_EI, type="o", col="orange", pch=18, lwd=2, cex=1.5,
     xlab="Iteration", ylab="Max EI (ParEGO)", main="", cex.main=1.5, cex.lab=1.3)
grid(lwd=1)

# Figure 4-2 (WDEI)
par(mfrow=c(1,1), mar=c(5,5,4,2))
plot(iter_seq, wdei_history$max_S, type="o", col="darkgreen", pch=20, lwd=2, cex=1.5,
     xlab="Iteration", ylab="Max Rank Score", main="", cex.main=1.5, cex.lab=1.3)
grid(lwd=1)

# --- Plot 3: Convergence of WDEI ---
par(mfrow=c(1,1), mar=c(5,5,4,2))
plot(iter_seq, wdei_history$max_WDEI, 
     type="o", col="purple", pch=20, lwd=2, cex=1.5,
     xlab="Iteration", ylab="Max Integral EI (WDEI)",
     main="", cex.main=1.5, cex.lab=1.3)
grid(lwd=1)

# Figure 5
par(mfrow=c(2,2), mar=c(4,4,3,1))
draw_opt_plot <- function(pdata, title) {
  ylim_r <- range(c(pdata$Min, pdata$True, pdata$Pred + 2*pdata$SD, pdata$Pred - 2*pdata$SD))
  plot(tt, pdata$Min, type='l', lwd=2, col='black', ylim=ylim_r, xlab="Time", ylab="Functional response", main=title, cex.main=1.1)
  polygon(c(tt, rev(tt)), c(pdata$Pred + 1.96*pdata$SD, rev(pdata$Pred - 1.96*pdata$SD)), 
          col=adjustcolor("red", alpha.f=0.15), border=NA)
  lines(tt, pdata$Pred, lwd=2, col='red', lty=2); lines(tt, pdata$True, lwd=2, col='blue', lty=3)
  legend("topright", legend=c("Theoretical Min", "Predicted Min", "True Min", "95% PI"), 
         col=c("black","red", "blue", NA), lty=c(1,2,3,NA), lwd=c(2,2,2,NA), 
         fill=c(NA,NA,NA,adjustcolor("red", alpha.f=0.15)), border=NA, cex=0.8, bty="n")
}
draw_opt_plot(m1$Plot, "")
draw_opt_plot(m2$Plot, "")
draw_opt_plot(m3$Plot, "")
draw_opt_plot(m4$Plot, "")
par(mfrow=c(1,1))