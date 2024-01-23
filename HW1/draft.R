library(dplyr)

# 设置随机数种子
set.seed(123)

# 参数设置
u1 = 1
sigma1 = 1/3
u2 = -1
sigma2 = 1/5
pi_1 = 1/3
pi_2 = 2/3
n = 50

# 生成数据
uniform_random = runif(n, 0, 1)
data_points = (uniform_random <= pi_1) * rnorm(n, u1, sigma1) + (uniform_random > pi_1) * rnorm(n, u2, sigma2)

# 损失函数
loss_function = function(x, data_points = data_points) {
  n = length(data_points)
  k = length(x) / 3
  loss = 0
  
  pi_value = exp(x[1:k]) / sum(exp(x[1:k]))
  
  density_list = list()
  for (i in 1:k) {
    density_list[[i]] = dnorm(data_points, x[k + i], x[2 * k + i])
  }
  density_matrix = data.frame(density_list) %>% as.matrix()
  
  loss = (density_matrix %*% pi_value) + 1e-5  # 添加小数以增加数值稳定性
  
  loss = loss %>% log() %>% sum()
  
  return(-loss)
}

# 梯度函数
grad_function <- function(x, data_points) {
  n <- length(data_points)
  k <- length(x) / 3
  loss <- 0
  
  pi_value <- exp(x[1:k]) / sum(exp(x[1:k]))
  
  exp_gamma_density_list <- list() # exp(gamma_i) * density_i matrix used in calculating all gradients
  
  centered_data_point_list <- list() # (x_i - u_i) / sigma_i^2 matrix used in calculating gradient of u
  
  second_order_centered_data_point_list <- list() # (x_i - u_i)^2 / sigma_i^3 matrix used in calculating gradient of sigma
  
  sum_exp_gamma_density <- rep(0, n)
  
  for (i in 1:k) {
    exp_gamma_density_value <- dnorm(data_points, x[k + i], x[2 * k + i]) * exp(x[i])
    
    exp_gamma_density_list[[i]] <- exp_gamma_density_value
    
    sum_exp_gamma_density <- sum_exp_gamma_density + exp_gamma_density_value + 1e-5
    
    centered_data_point_list[[i]] <- (data_points - x[k + i]) / (x[2 * k + i])
    
    second_order_centered_data_point_list[[i]] <- (data_points - x[k + i]) ^ 2 / (x[2 * k + i] ^ 3) - 1 / (x[2 * k + i])
    
  }
  
  exp_gamma_density_matrix <- data.frame(exp_gamma_density_list) %>% as.matrix()
  
  exp_gamma_density_matrix <- exp_gamma_density_matrix / sum_exp_gamma_density
  
  centered_data_point_matrix <- data.frame(centered_data_point_list) %>% as.matrix()
  
  second_order_centered_data_point_matrix <- data.frame(second_order_centered_data_point_list) %>% as.matrix()
  
  
  # gradient of gamma
  grad_gamma <- -(colSums(exp_gamma_density_matrix) - pi_value * n)
  
  # gradient of u
  grad_u <- -((centered_data_point_matrix * exp_gamma_density_matrix) %>% colSums())
  
  # gradient of sigma
  grad_sigma <- - ((second_order_centered_data_point_matrix * exp_gamma_density_matrix) %>% colSums())
  
  
  return(c(grad_gamma, grad_u, grad_sigma))
}

# 梯度下降函数
gradient_descent <- function(x,
                             data_points,
                             step_size = 0.01,
                             max_iter = 1000,
                             tol = 1e-3,
                             report = TRUE) {
  iter <- 0
  x_old <- rep(0, length(x))
  k <- length(x) / 3
  loss_list = c()
  
  loss <- loss_function(x, data_points)
  
  loss_list = c(loss_list, loss)
  
  while (iter < max_iter) {
    iter <- iter + 1
    x_old = x
    x = x - step_size * grad_function(x, data_points)
    loss_new <- loss_function(x, data_points)
    loss_list = c(loss_list, loss_new)
    
    if (report) {
      print(paste("iter:", iter, "loss:", loss_new, sep = " "))
      print(paste("gamma", x[1:k], sep = " "))
      print(paste("pi:", exp(x[1:k]) / sum(exp(x[1:k])), sep = " "))
      print(paste("u:", x[(k + 1):2 * k], sep = " "))
      print(paste("sigma:", x[(2 * k + 1):(3 * k)], sep = " "))
      print("\n ")
    }
    
    if ((abs(loss_new-loss)) < tol) {
      break
    }
  }
  
  
  result_list = list(
    loss = loss_list,
    pi_value = (exp(x[1:k]) / sum(exp(x[1:k]))) %>% as.vector() ,
    u = x[(k + 1):2 * k] %>% as.vector(),
    sigma = x[(2 * k + 1):(3 * k)] %>% as.vector(),
    iter = iter
  )
  
  return(result_list)
}

# 运行梯度下降
result <- gradient_descent(c(1,3,1,1,2,4), data_points)



plot(result$loss, type='l')

result[2:5]
