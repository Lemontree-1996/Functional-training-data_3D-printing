



########################################## Import scalar input ##########################################
# Data for 3D printing case: https://github.com/Lemontree-1996/Functional-training-data_3D-printing.git
Scalar<-read.csv("C:/Users/dcf/Desktop/IISE-Functional response/3D case/3D data/train_FRGP.csv",header=T)
scalar<-Scalar[,2:4] 

##################################### Import functional input #####################################
# Setting the file path
folder_path <- "C:/Users/dcf/Desktop/IISE-Functional response/3D case/3D data/DCF-YZ"  # Replace with the actual folder path
# 
txt_data <- list()

# 
for (i in 1:15) {
  # 
  file_name <- paste0(folder_path, "/", i, ".txt")
  
  #
  txt_data[[i]] <- read.table(file_name, header = TRUE)
}

#
head(txt_data[[1]])

# Loading packages
library(triangle)

# Parameterizing the triangular distribution of nozzle temperatures
a <- 202
b <- 208
c <- 205

# 
nozzle_temperatures <- list()

# 
for (i in 1:15) {
  set.seed(i)
  n_rows <- nrow(txt_data[[i]])
  #
  temp_data <- round(rtriangle(n_rows, a = a, b = b, c = c), 1)
  #
  nozzle_temperatures[[i]] <- temp_data
}

# 
head(nozzle_temperatures[[1]])
#
max_cols <- 500  

#
temperature_matrix <- matrix(NA, nrow = 15, ncol = max_cols)

#
for (i in 1:15) {
  set.seed(i)
  n_rows <- min(nrow(txt_data[[i]]), max_cols)
  #
  temp_data <- round(rtriangle(max_cols, a = 202, b = 208, c = 205), 1)
  #
  temperature_matrix[i, 1:max_cols] <- temp_data
}
#
rownames(temperature_matrix) <- paste("Sample", 1:15)
#
colnames(temperature_matrix) <- paste("T", 1:max_cols, sep = "")
#
head(temperature_matrix)

##################################### Import functional response #####################################
#
max_cols <- 500  
# 
data_matrix <- matrix(NA, nrow = 15, ncol = max_cols)
#
for (i in 1:15) {
  #
  temp_data <- txt_data[[i]]
  #
  n_rows <- min(nrow(temp_data), max_cols)
  data_matrix[i, 1:n_rows] <- temp_data[1:n_rows, 1]
}
#
rownames(data_matrix) <- paste("Sample", 1:15)
#
colnames(data_matrix) <- paste("T", 1:max_cols, sep = "")
#
data_matrix







########################################################### Modelling ###########################################################
#
library(GPFDA)
library(MASS)
library(lhs)
library(doParallel)
library(foreach)

set.seed(2025)

#
tt <- seq(0.04, 20, by = 0.04)
#
scalar_train <- as.matrix(scalar, ncol = 3)
x_train <- temperature_matrix
response_train <- data_matrix


########################################## 40 indexes evenly spaced out of 500 points ##########################################
index_40 <- round(seq(1, length(tt), length.out = 40))
tt <- tt[index_40]
x_train <- x_train[,index_40]
response_train <- response_train[,index_40]
#
hp <- list('pow.ex.v' = log(10), 'pow.ex.w' = log(1), 'vv' = log(1))
#
uCoefList <- list(
  list(lambda = 0.01, nbasi = 23),
  list(lambda = 0.01, nbasi = 23),
  list(lambda = 0.01, nbasi = 23)
)

gpfr_model <- gpfr(
  response = response_train, time = tt, uReg = scalar_train,
  fxReg = NULL, gpReg = x_train,
  fyList = list(nbasis = 23, lambda = 0.01),
  uCoefList = uCoefList,
  Cov = 'pow.ex', gamma = 1, fitting = TRUE
)

first_gpfr_model <- gpfr_model
first_gpfr_model_I <- gpfr_model

############################ Testing data ############################
#
M_pre <- 2500

#
param_ranges <- list(
  layer_thickness = c(0.1, 0.3),
  infilling_rate = c(0.1, 0.3),
  printing_speed = c(60, 80)
)
set.seed(20250225)
#
lhs_sample <- improvedLHS(M_pre, length(param_ranges))
#
layer_thickness_test <- lhs_sample[, 1] * diff(param_ranges$layer_thickness) + param_ranges$layer_thickness[1]
infilling_rate_test <- lhs_sample[, 2] * diff(param_ranges$infilling_rate) + param_ranges$infilling_rate[1]
printing_speed_test <- lhs_sample[, 3] * diff(param_ranges$printing_speed) + param_ranges$printing_speed[1]
#
scalar_test <- cbind(layer_thickness_test, infilling_rate_test, printing_speed_test)
#
x_test <- matrix(NA, M_pre, length(tt))
for (i in 1:M_pre) {
  set.seed(i)
  x_test[i, ] <- round(rtriangle(length(tt), a = 202, b = 208, c = 205), 1)
}

#
response_test <- matrix(NA, M_pre, length(tt))
for (i in 1:M_pre) {
  response_test[i, ] <- gpfrPredict(
    train = gpfr_model,
    testInputGP = x_test[i, ],
    testTime = tt,
    uReg = scalar_test[i, ],
    fxReg = NULL, gpReg = NULL
  )$ypred.mean
}


########################################################################### WDEI ###########################################
#
start_time1 <- Sys.time()
#
apply_weight_decay <- function(ei_matrix, history_indices, weight_adjustments, decay_factor = 0.5) {
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
adjust_ei_with_diversity <- function(ei_matrix, scalar_test, visited_indices, existing_points, diversity_weight = 0.5) {
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
N <- 200

# 
adjusted_ei_list <- list()
# 
raw_ei_values_list <- list()
standardized_ei_values_list <- list()

library(doParallel)
library(foreach)

# 
num_cores <- detectCores() - 2 
cl <- makeCluster(num_cores)
registerDoParallel(cl)

# 
iteration_times <- numeric(N)
for (iteration in 1:N) {
  #
  start_time <- Sys.time()
  # Recalculating fmin
  fmin <- apply(response_train, 2, max) #Previously it was min, here it is max
  #
  predictions <- foreach(i = 1:M_pre, .packages = "GPFDA") %dopar% {
    gpfrPredict(
      train = gpfr_model,
      testInputGP = x_test[i, ],
      testTime = tt,
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
    d <- ypred_mean - fmin
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
  weight_decay_results <- apply_weight_decay(ei_matrix, visited_indices, weight_adjustments, decay_factor = 0.5)
  ei_matrix <- weight_decay_results$ei_matrix
  weight_adjustments <- weight_decay_results$weight_adjustments
  # 
  adjusted_ei <- adjust_ei_with_diversity(ei_matrix, scalar_test, visited_indices, existing_points, diversity_weight = 0.5)
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



######################################################### Figure 8 #########################################################
#
WDEI <- sapply(add_point_results, function(result) result$global_max_integral)

#
plot(1:length(WDEI), WDEI, type = "l", col = "purple", lwd = 1, 
     main = "", ylim = c(0.5,0.9),
     xlab = "Iteration", ylab = "Maximum WDEI")
#
points(1:length(WDEI), WDEI, col = "green", pch = 18, cex = 0.8)
# 
grid(nx = NULL, ny = NULL, col = "gray", lty = "dotted", lwd = 0.8)
#
EI_raw <- sapply(raw_ei_values_list, max)

plot(1:length(EI_raw), EI_raw, type = "l", col = "blue", lwd = 1, 
     #cex.lab = 1, cex.axis = 1,
     main = "", ylim = c(0,25),
     xlab = "Iteration", ylab = "Maximum S")
#
points(1:length(EI_raw), EI_raw, col = "tomato", pch = 20, cex = 0.8)
grid(nx = NULL, ny = NULL, col = "gray", lty = "dotted", lwd = 0.8)


############## 保留加点的模型 ##############
final_gpfr_model1 <- final_gpfr_model


###################################################### 15+200 Model ######################################################
#  True_200 <- final_gpfr_model
############################ Testing data ############################
# 
M_pre1 <- 2500

# 
param_ranges1 <- list(
  layer_thickness = c(0.1, 0.3),
  infilling_rate = c(0.1, 0.3),
  printing_speed = c(60, 80)
)
set.seed(20250226)
# 
lhs_sample1 <- improvedLHS(M_pre1, length(param_ranges1))

#
layer_thickness_test1 <- lhs_sample1[, 1] * diff(param_ranges1$layer_thickness) + param_ranges1$layer_thickness[1]
infilling_rate_test1 <- lhs_sample1[, 2] * diff(param_ranges1$infilling_rate) + param_ranges1$infilling_rate[1]
printing_speed_test1 <- lhs_sample1[, 3] * diff(param_ranges1$printing_speed) + param_ranges1$printing_speed[1]

# 
scalar_test1 <- cbind(layer_thickness_test1, infilling_rate_test1, printing_speed_test1)



########################################################## Testing data ##########################################################
#
layer_thickness_test1 <- seq(0.1, 0.3, length.out = 21)
infilling_rate_test1  <- seq(0.1, 0.3, length.out = 21)
printing_speed_test1  <- seq(60, 80, length.out = 21)

#
scalar_test1 <- expand.grid(
  layer_thickness = layer_thickness_test1,
  infilling_rate = infilling_rate_test1,
  printing_speed = printing_speed_test1
)
M_pre1 <- nrow(scalar_test1)
scalar_test1 <- as.matrix(scalar_test1)
########################################################## Functional input data ##########################################################
#
x_test1 <- matrix(NA, M_pre1, length(tt))
for (i in 1:M_pre1) {
  set.seed(i)
  x_test1[i, ] <- round(rtriangle(length(tt), a = 202, b = 208, c = 205), 1)
}

#
response_test1 <- matrix(NA, M_pre1, length(tt))
for (i in 1:M_pre1) {
  response_test1[i, ] <- gpfrPredict(
    train = True_200,
    testInputGP = x_test1[i, ],
    testTime = tt,
    uReg = scalar_test1[i, ],
    fxReg = NULL, gpReg = NULL
  )$ypred.mean
}

########################################################################### Method 1 ###########################################
#
predictions_first <- list()
n_test <- length(tt);M_pre <- M_pre1
# Type II prediction
for (i in 1:M_pre) {
  predictions_first[[i]] <- gpfrPredict(
    train = first_gpfr_model,
    testInputGP = x_test1[i, ],
    testTime = tt,
    uReg = scalar_test1[i, ],
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

##################################################### Indicator #####################################################
#
rmse_values <- numeric(length = M_pre)
r_squared_values <- numeric(M_pre)
#ise_values <- numeric(M_pre)

#
for (i in 1:M_pre) {
  true_curve <- response_test1[i, ]
  predicted_curve <- predicted_means[i, ]
  
  # 1. RMSE
  rmse_values[i] <- sqrt(mean((predicted_curve - true_curve)^2))
  
  # 2. R-squared
  ss_total <- sum((true_curve - mean(true_curve))^2)
  ss_residual <- sum((true_curve - predicted_curve)^2)
  r_squared_values[i] <- 1 - (ss_residual / ss_total)
}

# 
rmse_mean <- mean(rmse_values)
r_squared_mean <- mean(r_squared_values)

# Printing results
summary_results <- data.frame(
  Metric = c("RMSE", "R-squared"),
  Mean = c(rmse_mean, r_squared_mean)
)



########################################################################### Method 3 ###########################################
# 
predictions_final <- list()
# Type II prediction
M_pre <- M_pre1
for (i in 1:M_pre) {
  predictions_final[[i]] <- gpfrPredict(
    train = final_gpfr_model1,
    testInputGP = x_test1[i, ],
    testTime = tt,
    uReg = scalar_test1[i, ],
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

###################### Indicator ########################
#
rmse_values1 <- numeric(length = M_pre)
r_squared_values1 <- numeric(M_pre)

#
for (i in 1:M_pre) {
  true_curve <- response_test1[i, ]
  predicted_curve <- predicted_means1[i, ]
  
  # 1. RMSE
  rmse_values1[i] <- sqrt(mean((predicted_curve - true_curve)^2))
  
  # 2. R-squared
  ss_total1 <- sum((true_curve - mean(true_curve))^2)
  ss_residual1 <- sum((true_curve - predicted_curve)^2)
  r_squared_values1[i] <- 1 - (ss_residual1 / ss_total1)
}

#
rmse_mean1 <- mean(rmse_values1)
r_squared_mean1 <- mean(r_squared_values1)


# Printing results
summary_results1 <- data.frame(
  Metric = c("RMSE", "R-squared"),
  Mean = c(rmse_mean1, r_squared_mean1)
)




########################################################################### Method 2 ###########################################
set.seed(20250226)
#
M <- 55
random_numbers <- sample(1:M_pre1, M)
u1_M <- scalar_test1[random_numbers,]

#
scalar_train_N <- rbind(scalar_train[1:15,], u1_M)

#
x_N <- rbind(x_train[1:15,], x_test1[random_numbers,])
response_N <- rbind(response_train[1:15,], response_test1[random_numbers,])

################################################## Randomly selected 70 sets + True_200 prediction 
set.seed(20250226)
#
lhs_sample_N <- improvedLHS(70, length(param_ranges))

#
layer_thickness_N <- lhs_sample_N[, 1] * diff(param_ranges$layer_thickness) + param_ranges$layer_thickness[1]
infilling_rate_N <- lhs_sample_N[, 2] * diff(param_ranges$infilling_rate) + param_ranges$infilling_rate[1]
printing_speed_N <- lhs_sample_N[, 3] * diff(param_ranges$printing_speed) + param_ranges$printing_speed[1]

#
scalar_train_N <- cbind(layer_thickness_N, infilling_rate_N, printing_speed_N)

#
x_N <- matrix(NA, 70, length(tt))
for (i in 1:70) {
  set.seed(i)
  x_N[i, ] <- round(rtriangle(length(tt), a = 202, b = 208, c = 205), 1)
}

#
response_N <- matrix(NA, 70, length(tt))
for (i in 1:70) {
  response_N[i, ] <- gpfrPredict(
    train = True_200,
    testInputGP = x_N[i, ],
    testTime = tt,
    uReg = scalar_train_N[i, ],
    fxReg = NULL, gpReg = NULL
  )$ypred.mean
}
#######################################################################################


#  GPFR modeling
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
    testInputGP = x_test1[i, ],
    testTime = tt,
    uReg = scalar_test1[i, ],
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

######################### Indicator ####################
#
rmse_values2 <- numeric(length = M_pre)
r_squared_values2 <- numeric(M_pre)


#
for (i in 1:M_pre) {
  true_curve <- response_test1[i, ]
  predicted_curve <- predicted_means2[i, ]
  
  # 1. RMSE
  rmse_values2[i] <- sqrt(mean((predicted_curve - true_curve)^2))
  
  # 2. R-squared
  ss_total <- sum((true_curve - mean(true_curve))^2)
  ss_residual <- sum((true_curve - predicted_curve)^2)
  r_squared_values2[i] <- 1 - (ss_residual / ss_total)
}

#
rmse_mean2 <- mean(rmse_values2)
r_squared_mean2 <- mean(r_squared_values2)

# Printing results
summary_results2 <- data.frame(
  Metric = c("RMSE", "R-squared"),
  Mean = c(rmse_mean2, r_squared_mean2)
)


##################################################### Type II Results #####################################################
print(summary_results)
print(summary_results2)
print(summary_results1)




########################################################################### Type II Optimization - Figure 9 ###########################################
#
F_star <- apply(response_test1, 2, max)
# Setting the weights α and β
alpha <- 0.5
beta <- 0.5

############################ 15 sets
# Calculate the MSE of each curve with respect to the target curve
mse_values <- apply(predicted_means, 1, function(F_j) {
  mean((F_star - F_j)^2)
})
#
total_variance <- rowSums(predicted_sds^2)

#
objective_values <- alpha * mse_values + beta * total_variance

#
optimal_index <- which.min(objective_values)
J <- objective_values[optimal_index]
optimal_curve <- predicted_means[optimal_index, ]
optimal_params <- scalar_test1[optimal_index, ]


# Calculate 95% confidence interval
upper_bound <- optimal_curve + 1.96 * predicted_sds[optimal_index, ]
lower_bound <- optimal_curve - 1.96 * predicted_sds[optimal_index, ]

# Plotting the objective curve vs. the optimal curve
plot(seq(1, 40), F_star, type = "l", col = "red", lwd = 2,
     xlab = "Time (s)", ylab = "Force (kN)", ylim = c(0,6), main = "", axes = FALSE)
#
axis(1, at = seq(1, 40, length.out = 5), labels = c(0, 5, 10, 15, 20))
#
axis(2)
# 
box()

#
polygon(c(seq(1, 40), rev(seq(1, 40))), 
        c(upper_bound, rev(lower_bound)), 
        col = rgb(0.6, 0.3, 0.8, alpha = 0.25), border = NA)
#
lines(seq(1, 40), optimal_curve, col = "blue", lwd = 2, lty = 2)  
#
legend("bottomright", legend = c("Objective curve", "Optimal curve", "95% Prediction Interval"), cex = 0.8,
       col = c("red", "blue", rgb(0.6, 0.3, 0.8, alpha = 0.25)), lty = c(1, 2, NA), lwd = c(2, 2, NA), 
       fill = c(NA, NA, rgb(0.6, 0.3, 0.8, alpha = 0.25)), border = NA)
#
grid(nx = NULL, ny = NULL, col = "gray", lty = "dotted", lwd = 0.8)



############################ 70 sets
#
mse_values2 <- apply(predicted_means2, 1, function(F_j) {
  mean((F_star - F_j)^2)
})
#
total_variance2 <- rowSums(predicted_sds2^2)
#
objective_values2 <- alpha * mse_values2 + beta * total_variance2
#
optimal_index2 <- which.min(objective_values2)
J2 <- objective_values2[optimal_index2]
optimal_curve2 <- predicted_means2[optimal_index2, ]
optimal_params2 <- scalar_test1[optimal_index2, ]

#
upper_bound2 <- optimal_curve2 + 1.96 * predicted_sds2[optimal_index2, ]
lower_bound2 <- optimal_curve2 - 1.96 * predicted_sds2[optimal_index2, ]

#
plot(seq(1, 40), F_star, type = "l", col = "red", lwd = 2,
     xlab = "Time (s)", ylab = "Force (kN)", ylim = c(0,6), main = "", axes = FALSE)
#
axis(1, at = seq(1, 40, length.out = 5), labels = c(0, 5, 10, 15, 20))
#
axis(2)
#
box()
#
polygon(c(seq(1, 40), rev(seq(1, 40))), 
        c(upper_bound2, rev(lower_bound2)), 
        col = rgb(0.6, 0.3, 0.8, alpha = 0.25), border = NA)
#
lines(seq(1, 40), optimal_curve2, col = "blue", lwd = 2, lty = 2)  
#
legend("bottomright", legend = c("Objective curve", "Optimal curve", "95% Prediction Interval"), cex = 0.8,
       col = c("red", "blue", rgb(0.6, 0.3, 0.8, alpha = 0.25)), lty = c(1, 2, NA), lwd = c(2, 2, NA), 
       fill = c(NA, NA, rgb(0.6, 0.3, 0.8, alpha = 0.25)), border = NA)
#
grid(nx = NULL, ny = NULL, col = "gray", lty = "dotted", lwd = 0.8)



############################ 15+55
#
mse_values1 <- apply(predicted_means1, 1, function(F_j) {
  mean((F_star - F_j)^2)
})
#
total_variance1 <- rowSums(predicted_sds1^2)
# 
objective_values1 <- alpha * mse_values1 + beta * total_variance1
#
optimal_index1 <- which.min(objective_values1)
J1 <- objective_values1[optimal_index1]
optimal_curve1 <- predicted_means1[optimal_index1, ]
optimal_params1 <- scalar_test1[optimal_index1, ]


#
upper_bound1 <- optimal_curve1 + 1.96 * predicted_sds1[optimal_index1, ]
lower_bound1 <- optimal_curve1 - 1.96 * predicted_sds1[optimal_index1, ]
#
plot(seq(1, 40), F_star, type = "l", col = "red", lwd = 2,
     xlab = "Time (s)", ylab = "Force (kN)", ylim = c(0,6), main = "", axes = FALSE)
# 
axis(1, at = seq(1, 40, length.out = 5), labels = c(0, 5, 10, 15, 20))
#
axis(2)
#
box()

#
polygon(c(seq(1, 40), rev(seq(1, 40))), 
        c(upper_bound1, rev(lower_bound1)), 
        col = rgb(0.6, 0.3, 0.8, alpha = 0.25), border = NA)
#
lines(seq(1, 40), optimal_curve1, col = "blue", lwd = 2, lty = 2)  
#
legend("bottomright", legend = c("Objective curve", "Optimal curve", "95% Prediction Interval"), cex = 0.8,
       col = c("red", "blue", rgb(0.6, 0.3, 0.8, alpha = 0.25)), lty = c(1, 2, NA), lwd = c(2, 2, NA), 
       fill = c(NA, NA, rgb(0.6, 0.3, 0.8, alpha = 0.25)), border = NA)
#
grid(nx = NULL, ny = NULL, col = "gray", lty = "dotted", lwd = 0.8)



#################################### Results - Table 5
optimal_params;optimal_params2;optimal_params1
J;J2;J1

true_curve <- F_star
best_prediction_mean1 <- optimal_curve
best_prediction_mean2 <- optimal_curve2
best_prediction_mean3 <- optimal_curve1

best_prediction_sd1 <- predicted_sds[optimal_index, ]
best_prediction_sd2 <- predicted_sds2[optimal_index2, ]
best_prediction_sd3 <- predicted_sds1[optimal_index1, ]

############################## Type II Optimization Results
rmse_opt_1 <- sqrt(mean((best_prediction_mean1 - true_curve)^2))
rmse_opt_2 <- sqrt(mean((best_prediction_mean2 - true_curve)^2))
rmse_opt_3 <- sqrt(mean((best_prediction_mean3 - true_curve)^2))

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

CP1 <- calculate_coverage_probability(true_curve,best_prediction_mean1,best_prediction_sd1,confidence_level = 0.95)
CP2 <- calculate_coverage_probability(true_curve,best_prediction_mean2,best_prediction_sd2,confidence_level = 0.95)
CP3 <- calculate_coverage_probability(true_curve,best_prediction_mean3,best_prediction_sd3,confidence_level = 0.95)

# Printing results
Opt_Type_II <- data.frame(
  Method = c("1", "2", "3"),
  RMSE = c(rmse_opt_1, rmse_opt_2, rmse_opt_3),
  SD = c(sd_mean_opt1, sd_mean_opt2, sd_mean_opt3),
  CP = c(CP1, CP2, CP3)
)

print(Opt_Type_II)
