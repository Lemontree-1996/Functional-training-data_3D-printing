library(GPFDA)
require(MASS)

set.seed(20250602)

# 
M <- 20
n <- 60
p <- 2
hp <- list('pow.ex.v' = log(10), 'pow.ex.w' = log(1), 'vv' = log(1))


# 
tt <- seq(-3, 3, len = n)
a <- cos((tt)^2)
b <- sin((0.5 * tt)^3)

# 
library(lhs)

# 
param_ranges <- list(
  u0 = c(-1, 1),
  u1 = c(6, 8)
)

lhs_sample <- improvedLHS(M, length(param_ranges))

#
u0_train <- lhs_sample[, 1] * diff(param_ranges$u0) + param_ranges$u0[1]
u1_train <- lhs_sample[, 2] * diff(param_ranges$u1) + param_ranges$u1[1]


#
scalar_train <- cbind(u0_train, u1_train)
print(scalar_train)

# 
x_train <- matrix(NA, M, n)
response_train <- matrix(NA, M, n)

for (i in 1:M) {
  u0 <- scalar_train[i, 1]
  u1 <- scalar_train[i, 2]
  x <- exp(tt) + rnorm(n, 0, 0.1)
  Sigma <- cov.pow.ex(hyper = hp, input = x, gamma = 1)
  diag(Sigma) <- diag(Sigma) + exp(hp$vv)  #
  y <- u0 * a + u1 * b + mvrnorm(n = 1, mu = rep(0, n), Sigma = Sigma)
  scalar_train[i, ] <- c(u0, u1)
  x_train[i, ] <- x
  response_train[i, ] <- y
}
#
uCoefList <- list(
  list(lambda = 0.01, nbasi = 23),
  list(lambda = 0.01, nbasi = 23)
)

# GPFR
gpfr_model <- gpfr(
  response = response_train, time = tt, uReg = scalar_train,
  fxReg = NULL, gpReg = x_train,
  fyList = list(nbasis = 23, lambda = 0.01),
  uCoefList = uCoefList,
  Cov = 'pow.ex', gamma = 1, fitting = T
)

first_gpfr_model <- gpfr_model
first_gpfr_model_I <- gpfr_model

#
n_test <- n
t_test <- seq(-3, 3, len = n_test)
b_test <- sin((0.5 * t_test)^3)

#
u0_test <- seq(-1, 1, length.out = 51)
u1_test <- seq(6, 8, length.out = 51)

#
test_grid <- expand.grid(u0 = u0_test, u1 = u1_test)

#
M_pre <- nrow(test_grid)
scalar_test <- as.matrix(test_grid)

x_test <- matrix(NA, M_pre, n_test)
response_test <- matrix(NA, M_pre, n_test)

for (i in 1:M_pre) {
  u0 <- scalar_test[i, 1]
  u1 <- scalar_test[i, 2]
  x <- exp(tt) + rnorm(n, 0, 0.1)
  Sigma <- cov.pow.ex(hyper = hp, input = x, gamma = 1)
  diag(Sigma) <- diag(Sigma) + exp(hp$vv)
  y <- u0 * a + u1 * b_test + mvrnorm(n = 1, mu = rep(0, n_test), Sigma = Sigma)
  scalar_test[i, ] <- c(u0, u1)
  x_test[i, ] <- x
  response_test[i, ] <- y
}



########################################################################### WDEI ###########################################
# 
start_time1 <- Sys.time()
# 
apply_weight_decay <- function(ei_matrix, history_indices, weight_adjustments, decay_factor = 0.50) {
  for (index in history_indices) {
    weight_adjustments[index] <- weight_adjustments[index] * decay_factor
    ei_matrix[index, ] <- ei_matrix[index, ] * weight_adjustments[index]
  }
  return(list(ei_matrix = ei_matrix, weight_adjustments = weight_adjustments))
}

#
calculate_diversity <- function(new_point, existing_points) {
  distances <- apply(existing_points, 1, function(existing) sqrt(sum((new_point - existing)^2)))
  return(min(distances))
}

#
adjust_ei_with_diversity <- function(ei_matrix, scalar_test, visited_indices, existing_points, diversity_weight = 0.50) {
  #
  ei_values <- apply(ei_matrix, 1, sum)
  
  #
  normalized_ei <- (ei_values - min(ei_values)) / (max(ei_values) - min(ei_values) + 1e-8)
  
  #
  diversity_scores <- sapply(1:nrow(scalar_test), function(i) {
    if (i %in% visited_indices) return(0)
    calculate_diversity(scalar_test[i, ], existing_points)
  })
  
  #
  normalized_diversity <- (diversity_scores - min(diversity_scores)) / 
    (max(diversity_scores) - min(diversity_scores) + 1e-8)
  
  #
  adjusted_ei <- (1 - diversity_weight) * normalized_ei + diversity_weight * normalized_diversity
  return(adjusted_ei)
}

#
add_point_results <- list()
visited_indices <- c()
existing_points <- scalar_train
weight_adjustments <- rep(1, M_pre)

#
final_gpfr_model <- NULL
N <- 80


#
adjusted_ei_list <- list()
#
raw_ei_values_list <- list()
standardized_ei_values_list <- list()

library(doParallel)
library(foreach)

#
num_cores <- detectCores() - 1 
cl <- makeCluster(num_cores)
registerDoParallel(cl)

#
iteration_times <- numeric(N)

for (iteration in 1:N) {
  #
  start_time <- Sys.time()
  
  #
  fmin <- apply(response_train, 2, min)
  
  #
  predictions <- foreach(i = 1:M_pre, .packages = "GPFDA") %dopar% {
    gpfrPredict(
      train = gpfr_model,
      testInputGP = x_test[i, ],
      testTime = t_test,
      uReg = scalar_test[i, ],
      fxReg = NULL,
      gpReg = NULL
    )
  }
  
  #
  ei_matrix <- matrix(NA, nrow = length(predictions), ncol = length(fmin))
  for (i in 1:length(predictions)) {
    ypred_mean <- predictions[[i]]$ypred.mean
    ypred_sd <- predictions[[i]]$ypred.sd
    d <- fmin - ypred_mean
    dn <- d / ypred_sd
    ei_matrix[i, ] <- d * pnorm(dn) + ypred_sd * dnorm(dn)
  }
  
  #
  raw_ei_values <- apply(ei_matrix, 1, sum)
  raw_ei_values_list[[iteration]] <- raw_ei_values
  
  #
  standardized_ei <- (raw_ei_values - min(raw_ei_values)) / (max(raw_ei_values) - min(raw_ei_values) + 1e-8)
  standardized_ei_values_list[[iteration]] <- standardized_ei
  
  #
  weight_decay_results <- apply_weight_decay(ei_matrix, visited_indices, weight_adjustments, decay_factor = 0.50)
  ei_matrix <- weight_decay_results$ei_matrix
  weight_adjustments <- weight_decay_results$weight_adjustments
  
  #
  adjusted_ei <- adjust_ei_with_diversity(ei_matrix, scalar_test, visited_indices, existing_points, diversity_weight = 0.50)
  adjusted_ei_list[[iteration]] <- adjusted_ei
  
  #
  max_curve_index <- which.max(adjusted_ei)
  global_max_integral <- adjusted_ei[max_curve_index]
  
  #
  ei_curve_max <- ei_matrix[max_curve_index, ]
  max_time_index <- which.max(ei_curve_max)
  
  #
  selected_scalar_input <- scalar_test[max_curve_index, ]
  selected_functional_input <- x_test[max_curve_index, ]
  selected_scalar_response <- response_test[max_curve_index, ]
  
  scalar_train <- rbind(scalar_train, selected_scalar_input)
  x_train <- rbind(x_train, selected_functional_input)
  response_train <- rbind(response_train, selected_scalar_response)
  
  #
  gpfr_model <- gpfr(
    response = response_train, time = tt, uReg = scalar_train,
    fxReg = NULL, gpReg = x_train,
    fyList = list(nbasis = 23, lambda = 0.01),
    uCoefList = uCoefList,
    Cov = 'pow.ex', gamma = 1, fitting = TRUE
  )
  
  #
  final_gpfr_model <- gpfr_model
  
  #
  add_point_results[[iteration]] <- list(
    iteration = iteration,
    fmin = fmin,
    global_max_integral = global_max_integral,
    max_curve_index = max_curve_index,
    max_time_index = max_time_index,
    ei_curve_max = ei_curve_max,
    raw_ei_values = raw_ei_values,
    standardized_ei = standardized_ei,
    selected_scalar_input = selected_scalar_input,
    selected_functional_input = selected_functional_input,
    selected_scalar_response = selected_scalar_response
  )
  
  #
  print(paste("Iteration:", iteration))
  print(gpfr_model$hyper)
  cat("Log-Likelihood:", gpfr_model$logLik, "\n")
  
  #
  visited_indices <- c(visited_indices, max_curve_index)
  existing_points <- rbind(existing_points, selected_scalar_input)
  
  #
  end_time <- Sys.time()
  iteration_times[iteration] <- as.numeric(difftime(end_time, start_time, units = "secs"))
}

#
stopCluster(cl)

#
plot(1:N, iteration_times, type = "b", col = "green", pch = 19, lwd = 2,
     xlab = "Iteration", ylab = "Time (seconds)",
     main = "Computation Time Per Iteration")


# add_point_results
print(first_gpfr_model$hyper)
print(final_gpfr_model$hyper)
#
end_time1 <- Sys.time()
print(end_time1 - start_time1)





########################################################################### Model prediction performance before WDEI ###########################################
# 
predictions_first <- list()
# Type II prediction
for (i in 1:M_pre) {
  predictions_first[[i]] <- gpfrPredict(
    train = first_gpfr_model,
    testInputGP = x_test[i, ],
    testTime = t_test, 
    uReg = scalar_test[i, ],
    fxReg = NULL, 
    gpReg = NULL
  )
}
#
predicted_means <- matrix(NA, nrow = M_pre, ncol = n_test)
predicted_sds <- matrix(NA, nrow = M_pre, ncol = n_test)

#
for (i in 1:M_pre) {
  predicted_means[i, ] <- predictions_first[[i]]$ypred.mean
  predicted_sds[i, ] <- predictions_first[[i]]$ypred.sd
}

##################################################### Predictive metrics #####################################################
# 
rmse_values <- numeric(length = M_pre)
r_squared_values <- numeric(M_pre)

# 
for (i in 1:M_pre) {
  true_curve <- response_test[i, ]
  predicted_curve <- predicted_means[i, ]
  
  # 1. RMSE
  rmse_values[i] <- sqrt(mean((predicted_curve - true_curve)^2))
  
  # 2. R-squared
  ss_total <- sum((true_curve - mean(true_curve))^2)
  ss_residual <- sum((true_curve - predicted_curve)^2)
  r_squared_values[i] <- 1 - (ss_residual / ss_total)
  
}

#
rmse_mean <- mean(rmse_values);rmse_sd <- sd(rmse_values)
r_squared_mean <- mean(r_squared_values);r_squared_sd <- sd(r_squared_values)
#
summary_results <- data.frame(
  Metric = c("RMSE", "R-squared"),
  Mean = c(rmse_mean, r_squared_mean),
  SD = c(rmse_sd, r_squared_sd)
)


########################################################################### Model prediction performance after WDEI ###########################################
#
predictions_final <- list()
# Type II prediction
for (i in 1:M_pre) {
  predictions_final[[i]] <- gpfrPredict(
    train = final_gpfr_model,
    testInputGP = x_test[i, ],
    testTime = t_test,
    uReg = scalar_test[i, ],
    fxReg = NULL,
    gpReg = NULL
  )
}
#
predicted_means1 <- matrix(NA, nrow = M_pre, ncol = n_test)
predicted_sds1 <- matrix(NA, nrow = M_pre, ncol = n_test)

#
for (i in 1:M_pre) {
  predicted_means1[i, ] <- predictions_final[[i]]$ypred.mean
  predicted_sds1[i, ] <- predictions_final[[i]]$ypred.sd
}

###################### Predictive metrics ########################
#
rmse_values1 <- numeric(length = M_pre)
r_squared_values1 <- numeric(M_pre)
ise_values1 <- numeric(M_pre)

# 
for (i in 1:M_pre) {
  true_curve <- response_test[i, ]
  predicted_curve <- predicted_means1[i, ]
  
  # 1. RMSE
  rmse_values1[i] <- sqrt(mean((predicted_curve - true_curve)^2))
  
  # 2. R-squared
  ss_total1 <- sum((true_curve - mean(true_curve))^2)
  ss_residual1 <- sum((true_curve - predicted_curve)^2)
  r_squared_values1[i] <- 1 - (ss_residual1 / ss_total1)
  
}

#
rmse_mean1 <- mean(rmse_values1);rmse_sd1 <- sd(rmse_values1)
r_squared_mean1 <- mean(r_squared_values1);r_squared_sd1 <- sd(r_squared_values1)

#
summary_results1 <- data.frame(
  Metric = c("RMSE", "R-squared"),
  Mean = c(rmse_mean1, r_squared_mean1),
  SD = c(rmse_sd1, r_squared_sd1)
)


########################################################################### N - One shot ###########################################
set.seed(20250602)

#
M_N <- N+M
n <- 60
p <- 2
hp <- list('pow.ex.v' = log(10), 'pow.ex.w' = log(1), 'vv' = log(1))

#
tt <- seq(-3, 3, len = n)
b <- sin((0.5 * tt)^3)

#
library(lhs)

#
param_ranges <- list(
  u0 = c(-1, 1),
  u1 = c(6, 8)
)

#
lhs_sample_N <- improvedLHS(M_N, length(param_ranges))

# 
u0_N <- lhs_sample_N[, 1] * diff(param_ranges$u0) + param_ranges$u0[1]
u1_N <- lhs_sample_N[, 2] * diff(param_ranges$u1) + param_ranges$u1[1]

#
scalar_train_N <- cbind(u0_N, u1_N)
print(scalar_train_N)

#
x_N <- matrix(NA, M_N, n)
response_N <- matrix(NA, M_N, n)

for (i in 1:M_N) {
  u0 <- scalar_train_N[i, 1]
  u1 <- scalar_train_N[i, 2]
  x <- exp(tt) + rnorm(n, 0, 0.1)
  Sigma <- cov.pow.ex(hyper = hp, input = x, gamma = 1)
  diag(Sigma) <- diag(Sigma) + exp(hp$vv)
  y <- u0 * a + u1 * b + mvrnorm(n = 1, mu = rep(0, n), Sigma = Sigma)
  scalar_train_N[i, ] <- c(u0, u1)
  x_N[i, ] <- x
  response_N[i, ] <- y
}
#
uCoefList <- list(
  list(lambda = 0.01, nbasi = 23),
  list(lambda = 0.01, nbasi = 23)
)

#
gpfr_N <- gpfr(
  response = response_N, time = tt, uReg = scalar_train_N,
  fxReg = NULL, gpReg = x_N,
  fyList = list(nbasis = 23, lambda = 0.01),
  uCoefList = uCoefList,
  Cov = 'pow.ex', gamma = 1, fitting = T
)


#
predictions_N <- list()
# Type II prediction
for (i in 1:M_pre) {
  predictions_N[[i]] <- gpfrPredict(
    train = gpfr_N,
    testInputGP = x_test[i, ],
    testTime = t_test,
    uReg = scalar_test[i, ],
    fxReg = NULL,
    gpReg = NULL
  )
}
#
predicted_means2 <- matrix(NA, nrow = M_pre, ncol = n_test)
predicted_sds2 <- matrix(NA, nrow = M_pre, ncol = n_test)

#
for (i in 1:M_pre) {
  predicted_means2[i, ] <- predictions_N[[i]]$ypred.mean
  predicted_sds2[i, ] <- predictions_N[[i]]$ypred.sd
}

######################### Predictive metrics ####################
#
rmse_values2 <- numeric(length = M_pre)
r_squared_values2 <- numeric(M_pre)
ise_values2 <- numeric(M_pre)

#
for (i in 1:M_pre) {
  true_curve <- response_test[i, ]
  predicted_curve <- predicted_means2[i, ]
  
  # 1. RMSE
  rmse_values2[i] <- sqrt(mean((predicted_curve - true_curve)^2))
  
  # 2. R-squared
  ss_total <- sum((true_curve - mean(true_curve))^2)
  ss_residual <- sum((true_curve - predicted_curve)^2)
  r_squared_values2[i] <- 1 - (ss_residual / ss_total)

}

#
rmse_mean2 <- mean(rmse_values2);rmse_sd2 <- sd(rmse_values2)
r_squared_mean2 <- mean(r_squared_values2);r_squared_sd2 <- sd(r_squared_values2)

#
summary_results2 <- data.frame(
  Metric = c("RMSE", "R-squared"),
  Mean = c(rmse_mean2, r_squared_mean2),
  SD = c(rmse_sd2, r_squared_sd2)
)


##################################################### Type II Results #####################################################
print(summary_results)
print(summary_results2)
print(summary_results1)




#################################### Nature color ####################################
library(ggsci)
library("scales")
# ?scale_color_npg()
#
mycolor <- pal_npg("nrc", alpha = 0.25)(10)
# mycolor
# show_col(mycolor)

########################################################## Figure 4 (Type II) ##########################################################
######################### Figure 4 (a)
par(family = "serif")
#
par(mfrow = c(1, 1), mar = c(5, 5, 1, 1))
#
plot(u0_train, u1_train,
     main = "",
     xlab = "u0_train",
     ylab = "u1_train",
     pch = 19, 
     col = "blue",
     xlim = range(param_ranges$u0),
     ylim = range(param_ranges$u1)
)

#
grid(nx = NULL, ny = NULL, col = "gray", lty = "dotted", lwd = 0.8)

######################### Figure 4 (a1) N-WDEI
WDEI <- sapply(add_point_results, function(result) result$global_max_integral)
#
plot(1:length(WDEI), WDEI, type = "l", col = "purple", lwd = 1, 
     main = "", ylim = c(0.6,1),
     xlab = "Iteration", ylab = "Maximum WDEI")
#
points(1:length(WDEI), WDEI, col = "green", pch = 18, cex = 0.8)
#
grid(nx = NULL, ny = NULL, col = "gray", lty = "dotted", lwd = 0.8)

#Raw_EI
EI_raw <- sapply(raw_ei_values_list, max)
plot(1:length(EI_raw), EI_raw, type = "l", col = "blue", lwd = 1, 
     cex.lab = 1.6, cex.axis = 1.4,
     main = "", ylim = c(0,8),
     xlab = "Iteration", ylab = "Maximum S")
# 添加点
points(1:length(EI_raw), EI_raw, col = "tomato", pch = 20, cex = 0.8)
grid(nx = NULL, ny = NULL, col = "gray", lty = "dotted", lwd = 0.8)

######################### Figure 4 (b) 
#
plot(u0_train, u1_train,
     main = "",
     xlab = "u0_new",
     ylab = "u1_new",
     pch = 19,
     col = "blue",
     xlim = range(param_ranges$u0),
     ylim = range(param_ranges$u1)
)
#
grid(nx = NULL, ny = NULL, col = "gray", lty = "dotted", lwd = 0.8)
Infill_scalar_input <- t(sapply(add_point_results, function(result) result$selected_scalar_input))
#
points(Infill_scalar_input[,1], Infill_scalar_input[,2], col = "tomato", pch = 18, cex = 0.8)



##################################################### Type I Results #####################################################
########################################################################### Before WDEI ###########################################
#
set.seed(20250602)
predictions_first_I <- list()
# Type I prediction
for (i in 1:M_pre) {
  predictions_first_I[[i]] <- gpfrPredict(
    train = first_gpfr_model_I,
    testInputGP = x_test[i, ],
    testTime = t_test, 
    uReg = scalar_test[i, ],
    fxReg = NULL, 
    gpReg = list('response' = response_test[i, ],
                 'input' = x_test[i, ],
                 'time' = t_test)
  )
}
#
predicted_means_I <- matrix(NA, nrow = M_pre, ncol = n_test)
predicted_sds_I <- matrix(NA, nrow = M_pre, ncol = n_test)

#
for (i in 1:M_pre) {
  predicted_means_I[i, ] <- predictions_first_I[[i]]$ypred.mean
  predicted_sds_I[i, ] <- predictions_first_I[[i]]$ypred.sd
}

################### Predictive metrics ##################
#
rmse_values_I <- numeric(length = M_pre)
r_squared_values_I <- numeric(M_pre)
ise_values_I <- numeric(M_pre)

#
for (i in 1:M_pre) {
  true_curve <- response_test[i, ]
  predicted_curve <- predicted_means_I[i, ]
  #
  # 1. RMSE
  rmse_values_I[i] <- sqrt(mean((predicted_curve - true_curve)^2))
  
  # 2. R-squared
  ss_total <- sum((true_curve - mean(true_curve))^2)
  ss_residual <- sum((true_curve - predicted_curve)^2)
  r_squared_values_I[i] <- 1 - (ss_residual / ss_total)

}

#
rmse_mean_I <- mean(rmse_values_I);rmse_sd_I <- sd(rmse_values_I)
r_squared_mean_I <- mean(r_squared_values_I);r_squared_sd_I <- sd(r_squared_values_I)

#
summary_results_I <- data.frame(
  Metric = c("RMSE", "R-squared"),
  Mean = c(rmse_mean_I, r_squared_mean_I),
  SD = c(rmse_sd_I, r_squared_sd_I)
)



########################################################################### N-One shot ###########################################
#
predictions_N_I <- list()
# Type I prediction
for (i in 1:M_pre) {
  predictions_N_I[[i]] <- gpfrPredict(
    train = gpfr_N,
    testInputGP = x_test[i, ],
    testTime = t_test,
    uReg = scalar_test[i, ],
    fxReg = NULL,
    gpReg = list('response' = response_test[i, ],
                 'input' = x_test[i, ],
                 'time' = t_test)
  )
}
#
predicted_means2_I <- matrix(NA, nrow = M_pre, ncol = n_test)
predicted_sds2_I <- matrix(NA, nrow = M_pre, ncol = n_test)

#
for (i in 1:M_pre) {
  predicted_means2_I[i, ] <- predictions_N_I[[i]]$ypred.mean
  predicted_sds2_I[i, ] <- predictions_N_I[[i]]$ypred.sd
}

################## Predictive metrics ##################
#
rmse_values2_I <- numeric(length = M_pre)
r_squared_values2_I <- numeric(M_pre)
ise_values2_I <- numeric(M_pre)

#
for (i in 1:M_pre) {
  true_curve <- response_test[i, ]
  predicted_curve <- predicted_means2_I[i, ]
  
  # 1. RMSE
  rmse_values2_I[i] <- sqrt(mean((predicted_curve - true_curve)^2))
  
  # 2. R-squared
  ss_total <- sum((true_curve - mean(true_curve))^2)
  ss_residual <- sum((true_curve - predicted_curve)^2)
  r_squared_values2_I[i] <- 1 - (ss_residual / ss_total)
 
}

# 
rmse_mean2_I <- mean(rmse_values2_I);rmse_sd2_I <- sd(rmse_values2_I)
r_squared_mean2_I <- mean(r_squared_values2_I);r_squared_sd2_I <- sd(r_squared_values2_I)
#
summary_results2_I <- data.frame(
  Metric = c("RMSE", "R-squared"),
  Mean = c(rmse_mean2_I, r_squared_mean2_I),
  SD = c(rmse_sd2_I, r_squared_sd2_I)
)


########################################################################### After WDEI ###########################################
#
predictions_final_I <- list()
# Type I prediction
for (i in 1:M_pre) {
  predictions_final_I[[i]] <- gpfrPredict(
    train = final_gpfr_model,
    testInputGP = x_test[i, ],
    testTime = t_test,
    uReg = scalar_test[i, ],
    fxReg = NULL,
    gpReg = list('response' = response_test[i, ],
                 'input' = x_test[i, ],
                 'time' = t_test)
  )
}
#
predicted_means1_I <- matrix(NA, nrow = M_pre, ncol = n_test)
predicted_sds1_I <- matrix(NA, nrow = M_pre, ncol = n_test)

#
for (i in 1:M_pre) {
  predicted_means1_I[i, ] <- predictions_final_I[[i]]$ypred.mean
  predicted_sds1_I[i, ] <- predictions_final_I[[i]]$ypred.sd
}

###################### Predictive metrics #####################
#
rmse_values1_I <- numeric(length = M_pre)
r_squared_values1_I <- numeric(M_pre)
ise_values1_I <- numeric(M_pre)

#
for (i in 1:M_pre) {
  true_curve <- response_test[i, ]
  predicted_curve <- predicted_means1_I[i, ]
 
  # 1. RMSE
  rmse_values1_I[i] <- sqrt(mean((predicted_curve - true_curve)^2))
  
  # 2. R-squared
  ss_total1 <- sum((true_curve - mean(true_curve))^2)
  ss_residual1 <- sum((true_curve - predicted_curve)^2)
  r_squared_values1_I[i] <- 1 - (ss_residual1 / ss_total1)

}

#
rmse_mean1_I <- mean(rmse_values1_I);rmse_sd1_I <- sd(rmse_values1_I)
r_squared_mean1_I <- mean(r_squared_values1_I);r_squared_sd1_I <- sd(r_squared_values1_I)

# 
summary_results1_I <- data.frame(
  Metric = c("RMSE", "R-squared"),
  Mean = c(rmse_mean1_I, r_squared_mean1_I),
  SD = c(rmse_sd1_I, r_squared_sd1_I)
)
######################################## Type I Table Results
print(summary_results_I)
print(summary_results2_I)
print(summary_results1_I)






####################################################### Optimization #######################################################
library(GA)
#
set.seed(20250602)
pop_size <- 50
max_gen <- 100
#
u_lower <- c(-1, 6)
u_upper <- c(1, 8)

#
noise_lower <- 0.01
noise_upper <- 0.15

#
lower_bounds <- c(u_lower, noise_lower)
upper_bounds <- c(u_upper, noise_upper)


########################################################################### Type II Optimization ###########################################
#
current_fmin_history <- list()
#
objective_function <- function(params, model, fmin, tt) {
  #
  u0 <- params[1]
  u1 <- params[2]
  #
  x <- exp(tt) + rnorm(length(tt), mean = 0, sd = 0.05)
  #
  prediction <- gpfrPredict(
    train = model,
    testInputGP = x,
    testTime = tt,
    uReg = cbind(u0, u1),
    fxReg = NULL,
    gpReg = NULL
  )
  #
  y_pred <- prediction$ypred.mean
  #
  current_fmin <- apply(rbind(fmin, y_pred), 2, min)
  # 
  current_fmin_history[[iteration]] <- current_fmin
  #
  objective_value <- sum((current_fmin - y_pred)^2)
  #
  return(objective_value)
}

fmin_opt <- rep(-10,length(tt))
######################################################### Before WDEI
#
ga_first <- ga(
  type = "real-valued",
  fitness = function(params) -objective_function(params, model = first_gpfr_model, fmin = fmin_opt, tt = tt),
  lower = lower_bounds,
  upper = upper_bounds,
  popSize = pop_size,
  maxiter = max_gen,
  pmutation = 0.1
)
#
plot(ga_first)

#
best_solution1 <- ga_first@solution
best_u1 <- best_solution1[1:2]
best_noise_level1 <- best_solution1[3]
#
best_x1 <- exp(tt) + rnorm(length(tt), mean = 0, sd = best_noise_level1)


#
best_prediction1 <- gpfrPredict(
  train = first_gpfr_model,
  testInputGP = best_x1,
  testTime = tt,
  uReg = best_u1,
  fxReg = NULL,
  gpReg = NULL
)

best_prediction_mean1 <- best_prediction1$ypred.mean
best_prediction_sd1 <- best_prediction1$ypred.sd

#
u0_best1 <- best_u1[1]
u1_best1 <- best_u1[2]

#
Sigma_best1 <- cov.pow.ex(hyper = hp, input = best_x1, gamma = 1)
diag(Sigma_best1) <- diag(Sigma_best1) + exp(hp$vv)

#
true_curve1 <- u0_best1 + u1_best1 * b_test + mvrnorm(n = 1, mu = rep(0, length(tt)), Sigma = Sigma_best1)

#
par(family = "serif")
# 
par(mfrow = c(1, 1), mar = c(5, 5, 1, 1))
#
plot(tt, best_prediction_mean1, type = "l", col = "purple", lwd = 2, ylim = c(-20,20),cex.lab = 1.4, cex.axis = 1.2,
     main = "",
     xlab = "t", ylab = "Functional response")

#
polygon(c(tt, rev(tt)), 
        c(best_prediction_mean1 + best_prediction_sd1, rev(best_prediction_mean1 - best_prediction_sd1)),
        col = "#8491B43F", border = NA)

#
if (exists("true_curve")) {
  lines(tt, true_curve1, col = "tomato", lwd = 2)
}

#
legend("topright", legend = c("Predicted mean", "Prediction interval", "True curve"),
       col = c("purple", "#8491B43F",  "tomato"), 
       lty = c(1, NA,  1), lwd = c(2, NA, 2),
       fill = c(NA, "#8491B43F", NA, NA), border = NA)




######################################################### N-One shot
#
ga_N <- ga(
  type = "real-valued",
  fitness = function(params) -objective_function(params, model = gpfr_N, fmin = fmin_opt, tt = tt),
  lower = lower_bounds,
  upper = upper_bounds,
  popSize = pop_size,
  maxiter = max_gen,
  pmutation = 0.1
)
#
plot(ga_N)

#
best_solution2 <- ga_N@solution
best_u2 <- best_solution2[1:2]
best_noise_level2 <- best_solution2[3]

#
best_x2 <- exp(tt) + rnorm(length(tt), mean = 0, sd = best_noise_level2)


#
best_prediction2 <- gpfrPredict(
  train = gpfr_N,
  testInputGP = best_x2,
  testTime = tt,
  uReg = best_u2,
  fxReg = NULL,
  gpReg = NULL
)

best_prediction_mean2 <- best_prediction2$ypred.mean
best_prediction_sd2 <- best_prediction2$ypred.sd

#
u0_best2 <- best_u2[1]
u1_best2 <- best_u2[2]

#
Sigma_best2 <- cov.pow.ex(hyper = hp, input = best_x2, gamma = 1)
diag(Sigma_best2) <- diag(Sigma_best2) + exp(hp$vv)

#
true_curve2 <- u0_best2 + u1_best2 * b_test + mvrnorm(n = 1, mu = rep(0, length(tt)), Sigma = Sigma_best2)


#
plot(tt, best_prediction_mean2, type = "l", col = "purple", lwd = 2, ylim = c(-20,20),cex.lab = 1.4, cex.axis = 1.2,
     main = "",
     xlab = "t", ylab = "Functional response")

#
polygon(c(tt, rev(tt)), 
        c(best_prediction_mean2 + best_prediction_sd2, rev(best_prediction_mean2 - best_prediction_sd2)),
        col = "#8491B43F", border = NA)

#
if (exists("true_curve")) {
  lines(tt, true_curve2, col = "tomato", lwd = 2)
}

#
legend("topright", legend = c("Predicted mean", "Prediction interval", "True curve"),
       col = c("purple", "#8491B43F",  "tomato"), 
       lty = c(1, NA,  1), lwd = c(2, NA,  2),
       fill = c(NA, "#8491B43F", NA, NA), border = NA)



######################################################### After WDEI
#
ga_final <- ga(
  type = "real-valued",
  fitness = function(params) -objective_function(params, model = final_gpfr_model, fmin = fmin_opt, tt = tt),
  lower = lower_bounds,
  upper = upper_bounds,
  popSize = pop_size,
  maxiter = max_gen,
  pmutation = 0.1
)

#
plot(ga_final)

#
best_solution3 <- ga_final@solution
best_u3 <- best_solution3[1:2]
best_noise_level3 <- best_solution3[3]

#
best_x3 <- exp(tt) + rnorm(length(tt), mean = 0, sd = best_noise_level3)

#
best_prediction3 <- gpfrPredict(
  train = final_gpfr_model,
  testInputGP = best_x3,
  testTime = tt,
  uReg = best_u3,
  fxReg = NULL,
  gpReg = NULL
)

best_prediction_mean3 <- best_prediction3$ypred.mean
best_prediction_sd3 <- best_prediction3$ypred.sd

#
u0_best3 <- best_u3[1]
u1_best3 <- best_u3[2]

#
Sigma_best3 <- cov.pow.ex(hyper = hp, input = best_x3, gamma = 1)
diag(Sigma_best3) <- diag(Sigma_best3) + exp(hp$vv)

#
true_curve3 <- u0_best3 + u1_best3 * b_test + mvrnorm(n = 1, mu = rep(0, length(tt)), Sigma = Sigma_best3)

#
plot(tt, best_prediction_mean3, type = "l", col = "purple", lwd = 2, ylim = c(-20,20),cex.lab = 1.4, cex.axis = 1.2,
     main = "",
     xlab = "t", ylab = "Functional response")

#
polygon(c(tt, rev(tt)), 
        c(best_prediction_mean3 + best_prediction_sd3, rev(best_prediction_mean3 - best_prediction_sd3)),
        col = "#8491B43F", border = NA)

#
if (exists("true_curve")) {
  lines(tt, true_curve3, col = "tomato", lwd = 2)
}

#
legend("topright", legend = c("Predicted mean", "Prediction interval", "True curve"),
       col = c("purple", "#8491B43F",  "tomato"), 
       lty = c(1, NA, 1), lwd = c(2, NA, 2),
       fill = c(NA, "#8491B43F", NA, NA), border = NA)


##################### Type II Optimization Results
rmse_opt_1 <- sqrt(mean((best_prediction_mean1 - true_curve1)^2))
rmse_opt_2 <- sqrt(mean((best_prediction_mean2 - true_curve2)^2))
rmse_opt_3 <- sqrt(mean((best_prediction_mean3 - true_curve3)^2))

sd_mean_opt1 <- mean(best_prediction_sd1)
sd_mean_opt2 <- mean(best_prediction_sd2)
sd_mean_opt3 <- mean(best_prediction_sd3)

# Coverage Probability, CP
calculate_coverage_probability <- function(actual, predicted_mean, predicted_sd, confidence_level = 0.95) {
  z <- qnorm((1 + confidence_level) / 2)
  lower_bound <- predicted_mean - z * predicted_sd
  upper_bound <- predicted_mean + z * predicted_sd
  coverage <- mean((actual >= lower_bound) & (actual <= upper_bound))
  return(coverage)
}

calculate_coverage_probability <- function(actual, predicted_mean, predicted_sd, confidence_level = 0.95) {
  #
  z <- qnorm((1 + confidence_level) / 2)  
  
  #
  lower_bound <- predicted_mean - z * predicted_sd
  upper_bound <- predicted_mean + z * predicted_sd
  
  #
  covered_points <- (actual >= lower_bound) & (actual <= upper_bound)
  
  #
  print(data.frame(
    Actual = actual,
    Predicted_Mean = predicted_mean,
    Lower_Bound = lower_bound,
    Upper_Bound = upper_bound,
    Covered = covered_points
  ))
  
  #
  coverage <- mean(covered_points)
  
  return(coverage)
}


CP1 <- calculate_coverage_probability(true_curve1,best_prediction_mean1,best_prediction_sd1,confidence_level = 0.95)
CP2 <- calculate_coverage_probability(true_curve2,best_prediction_mean2,best_prediction_sd2,confidence_level = 0.95)
CP3 <- calculate_coverage_probability(true_curve3,best_prediction_mean3,best_prediction_sd3,confidence_level = 0.95)

#
Opt_Type_II <- data.frame(
  Method = c("1", "2", "3"),
  RMSE = c(rmse_opt_1, rmse_opt_2, rmse_opt_3),
  SD = c(sd_mean_opt1, sd_mean_opt2, sd_mean_opt3),
  CP = c(CP1, CP2, CP3)
)

print(Opt_Type_II)


