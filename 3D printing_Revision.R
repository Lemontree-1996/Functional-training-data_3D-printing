###########################################################
# 1. Import data
###########################################################
rm(list=ls())
library(triangle)
library(GPFDA)
library(MASS)
library(lhs)
library(DiceKriging)

# Scalar input
Scalar <- read.csv("C:/Users/dcf/Desktop/IISE-Functional response/3D case/3D data/train_FRGP.csv", header=T)
scalar <- Scalar[,2:4] 

# Functional input
folder_path <- "C:/Users/dcf/Desktop/IISE-Functional response/3D case/3D data/DCF-YZ"
txt_data <- list()
for (i in 1:15) {
  file_name <- paste0(folder_path, "/", i, ".txt")
  if(file.exists(file_name)) {
    txt_data[[i]] <- read.table(file_name, header = TRUE)
  } else {
    txt_data[[i]] <- data.frame(Force = rep(0, 500))
  }
}

# temperature_matrix
max_cols <- 500  
temperature_matrix <- matrix(NA, nrow = 15, ncol = max_cols)
for (i in 1:15) {
  set.seed(i)
  temp_data <- round(rtriangle(max_cols, a = 203, b = 207, c = 205), 1)
  temperature_matrix[i, ] <- temp_data
}

#  data_matrix (Functional response)
data_matrix <- matrix(NA, nrow = 15, ncol = max_cols)
for (i in 1:15) {
  temp_data <- txt_data[[i]]
  n_rows <- min(nrow(temp_data), max_cols)
  data_matrix[i, 1:n_rows] <- temp_data[1:n_rows, 1]
  if(n_rows < max_cols) data_matrix[i, (n_rows+1):max_cols] <- data_matrix[i, n_rows]
}

###########################################################
# 2. Pre_Modelling
###########################################################
time_vec <- seq(0.04, 20, by = 0.04)
index_40 <- round(seq(1, length(time_vec), length.out = 40))
time_vec <- time_vec[index_40]

scalar_train_15 <- as.matrix(scalar[1:15,])
x_train_15 <- temperature_matrix[1:15, index_40]
response_train_15 <- data_matrix[1:15, index_40]

uCoefList <- list(
  list(lambda = 0.01, nbasis = 23),
  list(lambda = 0.01, nbasis = 23),
  list(lambda = 0.01, nbasis = 23)
)

# M1: Initial Model
cat("Fitting M1 (Initial Model)...\n")
mod_m1 <- gpfr(response = response_train_15, time = time_vec, uReg = scalar_train_15, 
               fxReg = NULL, gpReg = x_train_15, 
               fyList = list(nbasis = 23, lambda = 0.01), 
               uCoefList = uCoefList, Cov = 'pow.ex', fitting = TRUE)

###########################################################
# 3.  15 + 100 ground true model (True_200 Oracle) 
###########################################################
cat("\nStep 2: Building Oracle via 200 iterations (Using WDEI)...")
M_pre <- 3000
Add_P <- 15
set.seed(2026)
lhs_sample <- improvedLHS(M_pre, 3)
scalar_test <- cbind(lhs_sample[, 1]*0.2+0.1, lhs_sample[, 2]*0.2+0.1, lhs_sample[, 3]*20+60)

x_test <- matrix(NA, M_pre, 40)
for (i in 1:M_pre) {
  set.seed(i)
  x_test[i, ] <- round(rtriangle(40, a = 202, b = 208, c = 205), 1)
}

# True_100 Oracle
u_o <- scalar_train_15; x_o <- x_train_15; y_o <- response_train_15
for(iteration in 1:100) {
  mw_o <- gpfr(response = y_o, time = time_vec, uReg = u_o, 
               fxReg = NULL, gpReg = x_o, 
               fyList = list(nbasis=23, lambda=0.01), uCoefList = uCoefList, Cov='pow.ex', fitting=T)
  
  s_idx <- sample(1:M_pre, 100) 
  pm <- matrix(NA, 100, 40); ps <- matrix(NA, 100, 40)
  for(k in 1:100) {
    p <- gpfrPredict(train = mw_o, testInputGP = x_test[s_idx[k], ], testTime = time_vec, uReg = scalar_test[s_idx[k], ])
    pm[k,] <- as.numeric(p$ypred.mean); ps[k,] <- as.numeric(p$ypred.sd)
  }
  
  fmax_o <- apply(y_o, 2, max)
  d_vals_o <- sweep(pm, 2, fmax_o, "-")
  ei_o <- rowSums(d_vals_o * pnorm(d_vals_o/ps) + ps * dnorm(d_vals_o/ps))
  dist_o <- apply(scalar_test[s_idx,], 1, function(c) min(sqrt(rowSums(sweep(u_o, 2, c, "-")^2))))
  idx_in_sample <- which.max(0.5 * rank(ei_o) + 0.5 * rank(dist_o * rowMeans(ps)))
  idx <- s_idx[idx_in_sample]
  
  new_y <- as.numeric(gpfrPredict(mod_m1, x_test[idx,], time_vec, scalar_test[idx,])$ypred.mean)
  u_o <- rbind(u_o, scalar_test[idx,]); x_o <- rbind(x_o, x_test[idx,]); y_o <- rbind(y_o, new_y)
  if(iteration %% 50 == 0) cat("\nProgress:", iteration, "/200")
}

cat("\nFitting True_200 (Oracle Model)...\n")
True_200 <- gpfr(response = y_o, time = time_vec, uReg = u_o, gpReg = x_o, 
                 fyList = list(nbasis=30, lambda=1e-4), 
                 uCoefList = replicate(3, list(nbasis=30, lambda=1e-4), simplify=F), 
                 Cov = 'pow.ex', fitting = T)

#  3000 candidate
response_test_true <- matrix(NA, M_pre, 40)
for(i in 1:M_pre) response_test_true[i,] <- as.numeric(gpfrPredict(True_200, x_test[i,], time_vec, scalar_test[i,])$ypred.mean)

###########################################################
# 4.  21x21x21 Grid
###########################################################
cat("\nGenerating 21x21x21 Evaluation Grid...\n")
v1_grid <- seq(0.1, 0.3, length.out = 21)
v2_grid <- seq(0.1, 0.3, length.out = 21)
v3_grid <- seq(60, 80, length.out = 21)
grid_df <- expand.grid(V1 = v1_grid, V2 = v2_grid, V3 = v3_grid)
eval_scalar_grid <- as.matrix(grid_df)
colnames(eval_scalar_grid) <- c("V1", "V2", "V3")
N_grid <- nrow(eval_scalar_grid)

set.seed(2026) 
eval_func_grid <- matrix(round(rtriangle(N_grid * 40, a=202, b=208, c=205), 1), nrow=N_grid, ncol=40)

cat("Calculating God Truth on 9261 Grid (Please wait ~1 min)...\n")
response_grid_true <- matrix(NA, N_grid, 40)
for(i in 1:N_grid) {
  p_true <- gpfrPredict(True_200, eval_func_grid[i, ], time_vec, eval_scalar_grid[i, ])
  response_grid_true[i,] <- as.numeric(p_true$ypred.mean)
  if(i %% 2000 == 0) cat(sprintf("  Truth Progress: %d / %d\n", i, N_grid))
}
F_star_grid <- apply(response_grid_true, 2, max) 

eval_fn_grid <- function(model_name, model_obj) {
  cat(sprintf("\nEvaluating %s on 9261 Grid...\n", model_name))
  p_m <- matrix(NA, N_grid, 40); p_s <- matrix(NA, N_grid, 40)
  for(i in 1:N_grid) {
    res <- gpfrPredict(model_obj, eval_func_grid[i, ], time_vec, eval_scalar_grid[i, ])
    p_m[i,] <- as.numeric(res$ypred.mean)
    p_s[i,] <- as.numeric(res$ypred.sd)
  }
  
  v_obj <- 0.5 * apply(p_m, 1, function(y) mean((y - F_star_grid)^2)) + 0.5 * rowMeans(p_s^2)
  best_idx <- which.min(v_obj)
  
  rmse <- sqrt(mean((p_m[best_idx,] - response_grid_true[best_idx,])^2))
  pog_val <- sqrt(mean((p_m[best_idx,] - F_star_grid)^2))
  cp <- mean((response_grid_true[best_idx,]) >= (p_m[best_idx,]-1.96*p_s[best_idx,]) & 
               (response_grid_true[best_idx,]) <= (p_m[best_idx,]+1.96*p_s[best_idx,]))
  
  opt_params <- eval_scalar_grid[best_idx, ]
  
  return(list(V=v_obj[best_idx], RMSE=rmse, POG=pog_val, CP=cp, U=opt_params, P=p_m[best_idx,], S=p_s[best_idx,]))
}

res1 <- eval_fn_grid("M1 (Initial)", mod_m1)

###########################################################
# 5. M2, M3, M4 Modelling
###########################################################

# --- M2: One-Shot ---
cat("\nTraining M2 (One-Shot)...\n")
idx_m2 <- sample(1:M_pre, Add_P)
u_m2 <- rbind(scalar_train_15, scalar_test[idx_m2,]); x_m2 <- rbind(x_train_15, x_test[idx_m2,])
y_m2 <- rbind(response_train_15, response_test_true[idx_m2,])
mod_m2 <- gpfr(response = y_m2, time = time_vec, uReg = u_m2, gpReg = list(x_m2), 
               fyList = list(nbasis = 23, lambda = 0.01), uCoefList = uCoefList, Cov = 'pow.ex', fitting = TRUE)
res2 <- eval_fn_grid("M2", mod_m2)

# --- M3: ParEGO ---
cat("\nTraining M3 (ParEGO)...\n")
u_m3 <- scalar_train_15; x_m3 <- x_train_15; y_m3 <- response_train_15
parego_trace <- numeric(Add_P)
for(iter in 1:Add_P) {
  set.seed(iter)
  w <- runif(length(time_vec)); w <- w / sum(w)
  y_scal <- apply(y_m3, 1, function(row) min(w * row)) 
  train_df <- data.frame(u_m3); colnames(train_df) <- c("V1", "V2", "V3")
  km_mod <- km(design = train_df, response = y_scal, covtype = "matern5_2", control = list(trace = FALSE))
  test_df <- data.frame(scalar_test); colnames(test_df) <- c("V1", "V2", "V3")
  pred_k <- predict(km_mod, test_df, "UK")
  f_max <- max(y_scal)
  delta <- pred_k$mean - f_max
  ei_k <- delta * pnorm(delta / pred_k$sd) + pred_k$sd * dnorm(delta / pred_k$sd)
  idx <- which.max(ei_k)
  parego_trace[iter] <- max(ei_k)
  new_y <- matrix(response_test_true[idx, ], nrow = 1) 
  u_m3 <- rbind(u_m3, scalar_test[idx, ]); x_m3 <- rbind(x_m3, x_test[idx, ]); y_m3 <- rbind(y_m3, new_y) 
  if(iter %% 10 == 0) cat("ParEGO Iter:", iter, "\n")
}
mod_m3_final <- gpfr(response = y_m3, time = time_vec, uReg = u_m3, fxReg = NULL, gpReg = x_m3, 
                     fyList = list(nbasis = 23, lambda = 0.01), uCoefList = uCoefList, Cov = 'pow.ex', fitting = TRUE)
res3 <- eval_fn_grid("M3", mod_m3_final)

# --- M4: WDEI ---
cat("\nTraining M4 (WDEI)...\n")
u_m4 <- scalar_train_15; x_m4 <- x_train_15; y_m4 <- response_train_15
wdei_trace <- numeric(Add_P)
for(iter in 1:Add_P) {
  mw <- gpfr(response = y_m4, time = time_vec, uReg = u_m4, fxReg = NULL, gpReg = x_m4, 
             fyList = list(nbasis = 23, lambda = 0.01), uCoefList = uCoefList, Cov = 'pow.ex', fitting = TRUE)
  s_idx <- sample(1:M_pre, 2000)
  pm_w <- matrix(NA, 2000, 40); ps_w <- matrix(NA, 2000, 40)
  for(i in 1:2000) {
    p <- gpfrPredict(train = mw, testInputGP = x_test[s_idx[i], ], testTime = time_vec, uReg = scalar_test[s_idx[i], ])
    pm_w[i,] <- as.numeric(p$ypred.mean); ps_w[i,] <- as.numeric(p$ypred.sd)
  }
  fmax_w <- apply(y_m4, 2, max)
  d_vals <- sweep(pm_w, 2, fmax_w, "-") 
  ei_vals <- rowSums(d_vals * pnorm(d_vals/ps_w) + ps_w * dnorm(d_vals/ps_w))
  dist_w <- apply(scalar_test[s_idx,], 1, function(c) min(sqrt(rowSums(sweep(u_m4, 2, c, "-")^2))))
  idx_in_sample <- which.max(0.5 * rank(ei_vals) + 0.5 * rank(dist_w * rowMeans(ps_w)))
  idx <- s_idx[idx_in_sample]
  wdei_trace[iter] <- ei_vals[idx_in_sample]
  new_y_true <- matrix(response_test_true[idx, ], nrow = 1)
  u_m4 <- rbind(u_m4, scalar_test[idx, ]); x_m4 <- rbind(x_m4, x_test[idx, ]); y_m4 <- rbind(y_m4, new_y_true)
  if(iter %% 10 == 0) cat("WDEI Iter:", iter, "\n")
}
mod_m4 <- gpfr(response = y_m4, time = time_vec, uReg = u_m4, fxReg = NULL, gpReg = x_m4, 
               fyList = list(nbasis = 23, lambda = 0.01), uCoefList = uCoefList, Cov = 'pow.ex', fitting = TRUE)
res4 <- eval_fn_grid("M4", mod_m4)


# ==============================================================================
# ★ LOOCV ★
# ==============================================================================
cat("\n========================================================\n")
cat("Starting Leave-One-Out Cross Validation (LOOCV) for 15 Real Samples...\n")
cat("This evaluates the GLOBAL PREDICTIVE power of the final datasets.\n")
cat("========================================================\n")

calc_loocv <- function(model_name, u_all, x_all, y_all) {
  n_real <- 15 
  rmse_list <- numeric(n_real)
  r2_list <- numeric(n_real)
  
  cat(sprintf("Running LOOCV for %s: ", model_name))
  for(i in 1:n_real) {
   
    u_train_loo <- u_all[-i, , drop=FALSE]
    colnames(u_train_loo) <- c("V1", "V2", "V3") 
    x_train_loo <- x_all[-i, , drop=FALSE]
    y_train_loo <- y_all[-i, , drop=FALSE]
    
    
    u_test_loo <- u_all[i, ]
    x_test_loo <- x_all[i, ]
    y_test_loo <- y_all[i, ]
    
    
    mod_loo <- gpfr(response = y_train_loo, time = time_vec, uReg = u_train_loo, 
                    fxReg = NULL, gpReg = x_train_loo, 
                    fyList = list(nbasis = 23, lambda = 0.01), 
                    uCoefList = uCoefList, Cov = 'pow.ex', fitting = TRUE)
    
    pred <- gpfrPredict(train = mod_loo, testInputGP = x_test_loo, testTime = time_vec, uReg = u_test_loo)
    pred_mean <- as.numeric(pred$ypred.mean)
    
    rmse_list[i] <- sqrt(mean((pred_mean - y_test_loo)^2))
    ss_tot <- sum((y_test_loo - mean(y_test_loo))^2)
    ss_res <- sum((y_test_loo - pred_mean)^2)
    r2_list[i] <- 1 - (ss_res / ss_tot)
    cat(".")
  }
  cat(" Done!\n")
  return(list(LOOCV_RMSE = mean(rmse_list), LOOCV_R2 = mean(r2_list)))
}

#
u_m1 <- scalar_train_15; x_m1 <- x_train_15; y_m1 <- response_train_15
loocv_m1 <- calc_loocv("M1", u_m1, x_m1, y_m1)
loocv_m2 <- calc_loocv("M2", u_m2, x_m2, y_m2)
loocv_m3 <- calc_loocv("M3", u_m3, x_m3, y_m3)
loocv_m4 <- calc_loocv("M4", u_m4, x_m4, y_m4)


###########################################################
# 6. Table 6 
###########################################################
Table_Final <- data.frame(
  Method     = c("M1 (Initial)", "M2 (One-Shot)", "M3 (ParEGO)", "M4 (WDEI)"),
  # --- (Optimization Metrics) ---
  Opt_RMSE   = c(res1$RMSE, res2$RMSE, res3$RMSE, res4$RMSE),
  POG        = c(res1$POG, res2$POG, res3$POG, res4$POG),
  CP         = c(res1$CP, res2$CP, res3$CP, res4$CP),
  Opt_Thick  = c(res1$U[1], res2$U[1], res3$U[1], res4$U[1]), 
  Opt_Infill = c(res1$U[2], res2$U[2], res3$U[2], res4$U[2]), 
  Opt_Speed  = c(res1$U[3], res2$U[3], res3$U[3], res4$U[3]),
  # --- 预测指标 (Prediction Metrics from LOOCV) ---
  LOOCV_RMSE = c(loocv_m1$LOOCV_RMSE, loocv_m2$LOOCV_RMSE, loocv_m3$LOOCV_RMSE, loocv_m4$LOOCV_RMSE),
  LOOCV_R2   = c(loocv_m1$LOOCV_R2, loocv_m2$LOOCV_R2, loocv_m3$LOOCV_R2, loocv_m4$LOOCV_R2)
)

numeric_cols <- c("Opt_RMSE", "POG", "CP", "Opt_Thick", "Opt_Infill", "Opt_Speed", "LOOCV_RMSE", "LOOCV_R2")
Table_Final[, numeric_cols] <- round(Table_Final[, numeric_cols], 4)

cat("\n\n--- Table 4: Final Evaluation (Optimization & Prediction) ---\n")
print(Table_Final)

###########################################################
# Figure 7
###########################################################
par(mfrow=c(1,1), mar=c(4.5, 4.5, 2, 2), family="serif")
plot(1:Add_P, wdei_trace, type="b", pch=19, col="purple", 
     main="", xlab="Iteration", ylab="Max WDEI Value")
grid()

# Figure 8
par(mfrow=c(2,2), mar=c(4, 4.5, 2, 1), family="serif", oma=c(0,0,0,0)) 
# 
max_y <- max(c(F_star_grid, res1$P+1.96*res1$S, res2$P+1.96*res2$S, res3$P+1.96*res3$S, res4$P+1.96*res4$S)) * 1.1
y_lims <- c(0, max_y)

draw_result <- function(res, title) {
  plot(time_vec, F_star_grid, type="n", ylim=y_lims, xlab="Time (s)", ylab="Force (kN)", main=title)
  grid()
  polygon(c(time_vec, rev(time_vec)), c(res$P + 1.96*res$S, rev(res$P - 1.96*res$S)), 
          col=rgb(0.6,0.3,0.8,0.25), border=NA)
  lines(time_vec, F_star_grid, col="red", lwd=2)
  lines(time_vec, res$P, col="blue", lty=2, lwd=2)
  #
  legend("bottomright", legend=c("Objective curve", "Optimal curve", "95% PI"),
         col=c("red", "blue", rgb(0.6,0.3,0.8,0.5)), 
         lty=c(1, 2, NA), pch=c(NA, NA, 15), bty="n", cex=0.8)
}

#
draw_result(res1, "")
draw_result(res2, "")
draw_result(res3, "")
draw_result(res4, "")

# Inter
#  M4 (WDEI) over 9261 grid
p_m_wdei <- matrix(NA, N_grid, 40)
p_s_wdei <- matrix(NA, N_grid, 40)
for(i in 1:N_grid) {
  res <- gpfrPredict(mod_m4, eval_func_grid[i, ], time_vec, eval_scalar_grid[i, ])
  p_m_wdei[i,] <- as.numeric(res$ypred.mean)
  p_s_wdei[i,] <- as.numeric(res$ypred.sd)
}
