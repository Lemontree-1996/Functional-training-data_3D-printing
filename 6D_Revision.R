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
set.seed(2026) # 

# ==============================================================================
# 1. 6-D 
# ==============================================================================
NOISE_BASE <- 0.20     
n_time <- 40           
M_init <- 30           
N_total <- 90          
N_add <- N_total - M_init
MIN_DIST_TOL <- 1e-4   

# WDEI α
w_div <- 0.5     
w_ei <- 1 - w_div

tt <- seq(0, 2, len = n_time)
# 
phi <- list(
  p1 = 0.5*tt,
  p2 = cos(pi * tt),
  p3 = (tt)^2,
  p4 = 2*tt,
  p5 = exp(tt)
)

# ------------------------------------------------------------------------------
# 6-D 
# ------------------------------------------------------------------------------
generate_data_6d_complex <- function(u, n_points, type = "train") {
  # u: [0, 1]
  f1 <- sum(u^2) - 0.8 * cos(2 * pi * u[1] * u[2])
  f2 <- exp(-u[3]) * sin(2 * pi * u[4])
  f3 <- (u[5] - 0.5)^2 + u[6] * u[1]
  f4 <- 1.2 * cos(pi * u[2]) * cos(pi * u[5])
  f5 <- sqrt(u[1]*u[3] + 0.01) + u[4]^2
  
  y_true <- f1*phi$p1 + f2*phi$p2 + f3*phi$p3 + f4*phi$p4 + f5*phi$p5
  
  if(type == "train") {
    y_obs <- y_true + rnorm(n_points, 0, NOISE_BASE)
  } else {
    y_obs <- y_true 
  }
  return(list(x = tt, y_obs = as.vector(y_obs), y_true = as.vector(y_true)))
}

# ------------------------------------------------------------------------------
# 
# ------------------------------------------------------------------------------
cat("Plotting 6-D functional features and coefficient surface...\n")
par(mfrow=c(1,2), family = "serif", mar=c(4.5,4.5,2,1))

plot(tt, rep(0, n_time), type="n", ylim=c(0, 10), xlab="t", ylab="y(u, t)", main="")
cols <- rainbow(10)
for(i in 1:10) {
  lines(tt, generate_data_6d_complex(runif(6), n_time, "test")$y_true, col=cols[i], lwd=1.2)
}
#
u1_g <- seq(0, 1, length.out=50); u2_g <- seq(0, 1, length.out=50)
f1_s <- matrix(0, 50, 50)
for(i in 1:50) for(j in 1:50) f1_s[i,j] <- (u1_g[i]^2 + u2_g[j]^2) - 0.8 * cos(2 * pi * u1_g[i] * u2_g[j])
image(u1_g, u2_g, f1_s, col=terrain.colors(100), xlab="u1", ylab="u2", main="")
contour(u1_g, u2_g, f1_s, add=TRUE)

# ------------------------------------------------------------------------------
# 
# ------------------------------------------------------------------------------
cat("\nApproximating theoretical envelope...")
n_env <- 2500
env_sample <- randomLHS(n_env, 6)
y_env_mat <- matrix(NA, n_env, n_time)
for(i in 1:n_env) y_env_mat[i,] <- generate_data_6d_complex(env_sample[i,], n_time, "test")$y_true
true_min_curve <- apply(y_env_mat, 2, min)

# ==============================================================================
# 2. 
# ==============================================================================
M_pre <- 600
scalar_test <- randomLHS(M_pre, 6)
y_test_true <- matrix(NA, M_pre, n_time)
x_test_standard <- matrix(rep(tt, M_pre), M_pre, n_time, byrow = TRUE)
for (i in 1:M_pre) y_test_true[i,] <- generate_data_6d_complex(scalar_test[i,], n_time, "test")$y_true

# ==============================================================================
# 3. 
# ==============================================================================

# --- Method 1: Initial Design (M=30) ---
cat("\nStep 1: Initial Design...")
u_init <- improvedLHS(M_init, 6)
y_init <- matrix(0, M_init, n_time); x_init <- matrix(rep(tt, M_init), M_init, n_time, byrow = TRUE)
for(i in 1:M_init) y_init[i,] <- generate_data_6d_complex(u_init[i,], n_time, "train")$y_obs
model_init <- gpfr(response=y_init, time=tt, uReg=u_init, gpReg=x_init, Cov='pow.ex', fitting=TRUE)

# --- Method 2: One-Shot LHS (N=60) ---
cat("\nStep 2: One-Shot LHS (N=60)...")
u_os <- improvedLHS(N_total, 6)
y_os <- matrix(0, N_total, n_time); x_os <- matrix(rep(tt, N_total), N_total, n_time, byrow = TRUE)
for(i in 1:N_total) y_os[i,] <- generate_data_6d_complex(u_os[i,], n_time, "train")$y_obs
model_os <- gpfr(response=y_os, time=tt, uReg=u_os, gpReg=x_os, Cov='pow.ex', fitting=TRUE)

# --- Method 3: ParEGO ---
cat("\nStep 3: ParEGO Optimization...")
u_parego <- u_init; y_parego <- y_init
for(iter in 1:N_add) {
  y_norm <- scale(y_parego, center=apply(y_parego, 2, min), scale=apply(y_parego, 2, max)-apply(y_parego, 2, min)+1e-6)
  w <- runif(n_time); w <- w/sum(w)
  f_scalar <- apply(y_norm, 1, function(r) max(w * r) + 0.05 * sum(w * r))
  model_k <- km(design=as.data.frame(u_parego), response=f_scalar, covtype="matern5_2", control=list(trace=FALSE))
  pred <- predict(model_k, newdata=as.data.frame(scalar_test), type="UK")
  ei <- (min(f_scalar) - pred$mean) * pnorm((min(f_scalar) - pred$mean)/pred$sd) + pred$sd * dnorm((min(f_scalar) - pred$mean)/pred$sd)
  new_u <- scalar_test[which.max(ei),]
  u_parego <- rbind(u_parego, new_u); y_parego <- rbind(y_parego, generate_data_6d_complex(new_u, n_time, "train")$y_obs)
}
model_parego <- gpfr(response=y_parego, time=tt, uReg=u_parego, gpReg=matrix(rep(tt, N_total), N_total, n_time, byrow = TRUE), Cov='pow.ex', fitting=TRUE)

# --- Method 4: WDEI (Rank-Based & Uncertainty-Aware) ---
cat("\nStep 4: WDEI (Proposed)...")
u_wdei <- u_init; y_wdei <- y_init; x_wdei <- x_init; visited <- c()
wdei_history <- list(max_WDEI = numeric(N_add))

for(iter in 1:N_add) {
  model_w <- gpfr(response=y_wdei, time=tt, uReg=u_wdei, gpReg=x_wdei, Cov='pow.ex', fitting=TRUE)
  mu_t <- matrix(NA, M_pre, n_time); sd_t <- matrix(NA, M_pre, n_time)
  for(i in 1:M_pre) {
    p <- gpfrPredict(model_w, testInputGP=x_test_standard[i,], testTime=tt, uReg=scalar_test[i,])
    mu_t[i,] <- p$ypred.mean; sd_t[i,] <- p$ypred.sd
  }
  #Plugin-EI 
  mu_tr <- matrix(NA, nrow(u_wdei), n_time)
  for(k in 1:nrow(u_wdei)) mu_tr[k,] <- gpfrPredict(model_w, testInputGP=x_wdei[k,], testTime=tt, uReg=u_wdei[k,])$ypred.mean
  fmin_p <- apply(mu_tr, 2, min)
  
  d_mat <- t(fmin_p - t(mu_t)); dn_mat <- d_mat / sd_t
  ei_int <- rowSums(d_mat * pnorm(dn_mat) + sd_t * dnorm(dn_mat))
  dists <- apply(scalar_test, 1, function(cand) min(sqrt(rowSums(sweep(u_wdei, 2, cand)^2))))
  div_val <- dists * rowMeans(sd_t)
  
  score <- w_ei * (rank(ei_int)/M_pre) + w_div * (rank(div_val)/M_pre)
  score[visited] <- -1
  best_idx <- which.max(score); visited <- c(visited, best_idx)
  new_u <- scalar_test[best_idx,]; d_new <- generate_data_6d_complex(new_u, n_time, "train")
  wdei_history$max_WDEI[iter] <- ei_int[best_idx]
  u_wdei <- rbind(u_wdei, new_u); y_wdei <- rbind(y_wdei, d_new$y_obs); x_wdei <- rbind(x_wdei, d_new$x)
  if(iter %% 10 == 0) cat(iter, " ")
}
model_wdei <- gpfr(response=y_wdei, time=tt, uReg=u_wdei, gpReg=x_wdei, Cov='pow.ex', fitting=TRUE)

# ==============================================================================
# 4.
# ==============================================================================
get_metrics <- function(model) {
  p_m <- matrix(NA, M_pre, n_time); p_s <- matrix(NA, M_pre, n_time)
  for(i in 1:M_pre) {
    res <- gpfrPredict(model, testInputGP=x_test_standard[i,], testTime=tt, uReg=scalar_test[i,])
    p_m[i,] <- res$ypred.mean; p_s[i,] <- res$ypred.sd
  }
  rmse_t2 <- sqrt(mean((p_m - y_test_true)^2))
  r2_t2 <- 1 - sum((y_test_true - p_m)^2) / sum((y_test_true - mean(y_test_true))^2)
  b_idx <- which.min(rowSums(p_m))
  pog <- sqrt(mean((p_m[b_idx,] - true_min_curve)^2))
  o_rmse <- sqrt(mean((p_m[b_idx,] - y_test_true[b_idx,])^2))

  cp <- mean( (y_test_true[b_idx,] + rnorm(n_time, 0, NOISE_BASE)) >= (p_m[b_idx,] - 1.96*p_s[b_idx,]) & 
                (y_test_true[b_idx,] + rnorm(n_time, 0, NOISE_BASE)) <= (p_m[b_idx,] + 1.96*p_s[b_idx,]) )
  return(list(Pred = c(rmse_t2, r2_t2), Opt = c(pog, o_rmse, cp), Plot = list(Pred=p_m[b_idx,], True=y_test_true[b_idx,], Min=true_min_curve, SD=p_s[b_idx,])))
}

cat("\nCalculating metrics..."); m1 <- get_metrics(model_init); m2 <- get_metrics(model_os); m3 <- get_metrics(model_parego); m4 <- get_metrics(model_wdei)

# Table B1
df1 <- data.frame(Method = c("Initial(M=30)", "One-Shot(N=60)", "ParEGO", "WDEI"), RMSE = c(m1$Pred[1], m2$Pred[1], m3$Pred[1], m4$Pred[1]), R2 = c(m1$Pred[2], m2$Pred[2], m3$Pred[2], m4$Pred[2]))
df2 <- data.frame(Method = c("Initial(M=30)", "One-Shot(N=60)", "ParEGO", "WDEI"), POG = c(m1$Opt[1], m2$Opt[1], m3$Opt[1], m4$Opt[1]), OptRMSE = c(m1$Opt[2], m2$Opt[2], m3$Opt[2], m4$Opt[2]), CP = c(m1$Opt[3], m2$Opt[3], m3$Opt[3], m4$Opt[3]))
cat("\nTable 1: Prediction\n"); print(round(df1[,-1], 4)); cat("\nTable 2: Optimization\n"); print(round(df2[,-1], 4))

# ==============================================================================
# 5. Figure B2
# ==============================================================================
par(mfrow=c(1,2), family="serif", mar=c(4.5,4.5,2,1))
#
plot(1:N_add, wdei_history$max_WDEI, type="o", col="purple", pch=20, xlab="Iteration", ylab="Max WDEI", main="")
#
pd <- m4$Plot
#
y_limits <- range(c(pd$Min, pd$True, pd$Pred + 2*pd$SD, pd$Pred - 2*pd$SD), na.rm=TRUE)

plot(tt, pd$Min, type='l', lwd=2, col='black', ylim=y_limits, xlab="t", ylab="Functional response", main="")
#
polygon_col <- adjustcolor("red", alpha.f=0.15)
polygon(c(tt, rev(tt)), c(pd$Pred + 1.96*pd$SD, rev(pd$Pred - 1.96*pd$SD)), col=polygon_col, border=NA)
#
lines(tt, pd$Pred, lwd=2, col='red', lty=2)
lines(tt, pd$True, lwd=2, col='blue', lty=3)
#
legend("topright", 
       legend = c("Theoretical Min", "WDEI Predicted", "WDEI True", "95% PI"), 
       col = c("black", "red", "blue", polygon_col),  #
       lty = c(1, 2, 3, NA),                          #
       lwd = c(2, 2, 2, NA),                          # 
       pch = c(NA, NA, NA, 15),                       #
       pt.cex = 2,                                    #
       cex = 0.8, 
       bty = "n")