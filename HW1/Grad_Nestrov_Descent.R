#Nestrov Accleration


default_data_generation = function(n=100) {
  set.seed(123)
  
  u1 = 1
  sigma1 = 1/3
  
  u2=-1
  sigma2=1/5
  
  pi_1 = 1/3
  pi_2 = 2/3
  
  uniform_random = runif(n,0,1)
  
  data_points = (uniform_random<=pi_1) * rnorm(n, u1, sigma1) + (uniform_random>pi_1) * rnorm(n, u2, sigma2)
  
  data_points %>% density() %>% plot()
  
  data_points %>% hist(,breaks=50)
  
  return(data_points)
}


data_points = default_data_generation(n=1000)



loss_function = function(x, data_points = data_points) {
  n = length(data_points)
  k = length(x) / 3
  loss = 0
  
  max_gamma = max(x[1:k])
  
  pi_value = exp(x[1:k]-max_gamma) / sum(exp(x[1:k]-max_gamma))
  
  density_list = list()
  for (i in 1:k) {
    density_list[[i]] = dnorm(data_points, x[k + i], x[2 * k + i])
  }
  density_matrix = data.frame(density_list) %>% as.matrix()
  
  loss = (density_matrix %*% pi_value) + 1e-5
  
  loss = loss %>% log() %>% sum()
  
  return(-loss)
}


grad_function = function(x, data_points)
{
  n = length(data_points)
  k = length(x) / 3
  loss = 0
  
  max_gamma = max(x[1:k])
  
  pi_value = exp(x[1:k]-max_gamma) / sum(exp(x[1:k]-max_gamma))
  
  exp_gamma_density_list = list() # exp(gamma_i) * density_i matrix used in claculating all gradients
  
  centered_data_point_list = list() # (x_i - u_i) / sigma_i^2 matrix used in calculating gradient of u
  
  second_order_centered_data_point_list = list() # (x_i - u_i)^2 / sigma_i^3 matrix used in calculating gradient of sigma
  
  sum_exp_gamma_density = rep(0, n)
  
  for (i in 1:k) {
    #print(exp(x[i]))
    
    exp_gamma_density_value = dnorm(data_points, x[k + i], x[2 * k + i]) * exp(x[i]-max_gamma)
    
    exp_gamma_density_list[[i]] =
      exp_gamma_density_value
    
    sum_exp_gamma_density =
      sum_exp_gamma_density + exp_gamma_density_value + 1e-5
    
    centered_data_point_list[[i]] =
      (data_points - x[k + i]) / (x[2 * k + i])
    
    second_order_centered_data_point_list[[i]] =
      (data_points - x[k + i]) ^ 2 / (x[2 * k + i] ^ 3) - 1 / (x[2 * k + i])
    
  }
  
  # Transforming into matrix
  
  exp_gamma_density_matrix = data.frame(exp_gamma_density_list) %>% as.matrix()
  
  
  exp_gamma_density_matrix = exp_gamma_density_matrix / sum_exp_gamma_density
  
  centered_data_point_matrix = data.frame(centered_data_point_list) %>% as.matrix()
  
  second_order_centered_data_point_matrix = data.frame(second_order_centered_data_point_list) %>% as.matrix()
  
  # gradient of gamma
  grad_gamma = -(colSums(exp_gamma_density_matrix) - pi_value * n)
  
  #print(grad_gamma)
  
  # gradient of u
  grad_u = -(centered_data_point_matrix * exp_gamma_density_matrix) %>% colSums()
  
  # gradient of sigma
  grad_sigma = -(second_order_centered_data_point_matrix * exp_gamma_density_matrix) %>% colSums()
  
  
  return(c(grad_gamma, grad_u, grad_sigma))
}




gradient_BB_descent = function(x,
                               data_points,
                               max_iter = 1000,
                               tol = 1e-5,
                               loss_tol=1e-5,
                               report = FALSE) {
  iter = 0
  x_old = rep(0, length(x))
  k = length(x) / 3
  loss_list = c()
  grad_list = c()
  
  u_frame = data.frame()
  sigma_frame = data.frame()
  pi_frame = data.frame()
  
  loss_new = loss_function(x, data_points)
  
  if (is.nan(loss) || is.infinite(loss)) {
    stop("Initial loss is NaN or infinite")
  }
  
  loss_list = c(loss_list, loss)
  
  gradient_old = rep(0, length(x))
  x_old_2 = rep(0, length(x))
  
  u_frame=rbind(x[(k+1):(2*k)],u_frame)
  sigma_frame=rbind(x[(2*k+1):(3*k)],sigma_frame)
  pi_frame=rbind(exp(x[1:k])/sum(exp(x[1:k])),pi_frame)
  
  
  
  while (iter < max_iter) {
    iter = iter + 1
    x_old = x
    
    gradient = grad_function(x, data_points)
    
    if (any(is.nan(gradient)) || any(is.infinite(gradient))) {
      print(as.vector(x_old))
      stop("Gradient is NaN or infinite at iteration ", iter)
      #print(x_old)
    }
    
    step_size = as.vector((x_old - x_old_2) %*% (gradient - gradient_old) ) / sum((gradient - gradient_old) ^ 2)
    
    #step_size = step_size * max(1,log10(1 + iter))
    
    x = x - step_size * gradient 
    u_frame=rbind(x[(k+1):(2*k)],u_frame)
    sigma_frame=rbind(x[(2*k+1):(3*k)],sigma_frame)
    pi_frame=rbind(exp(x[1:k])/sum(exp(x[1:k])),pi_frame)
    
    x[(2*k+1):(3*k)]=abs(x[(2*k+1):(3*k)])
    
    loss_old = loss_new
    loss_new = loss_function(x, data_points)
    if (is.nan(loss_new) || is.infinite(loss_new)) {
      
      stop("Loss is NaN or infinite at iteration ", iter)
      
    }
    loss_list = c(loss_list, loss_new)
    grad_list = c(grad_list, max(abs(gradient)))
    
    if (report) {
      cat("iter:", iter, "loss:", loss_new, "\n")
      cat("pi:", exp(x[1:k]) / sum(exp(x[1:k])), "\n")
      cat("u:", x[(k + 1):2 * k] , "\n")
      cat("sigma:", x[(2 * k + 1):(3 * k)] , "\n")
    }
    
    # if (max(abs(x-x_old)) < tol || abs(loss_new - loss_old) < loss_tol) {
    #   break
    # }
    if (max(abs(x-x_old)) < tol ) {
      break
    }
    
    gradient_old = gradient
    x_old_2 = x_old
    
  }
  
  names(pi_frame) = paste0("pi_",1:k)
  names(u_frame) = paste0("u_",1:k)
  names(sigma_frame) = paste0("sigma_",1:k)
  
  result_list = list(
    loss = loss_list,
    pi_value = (exp(x[1:k]) / sum(exp(x[1:k]))) %>% as.vector() ,
    u = x[(k + 1):2 * k] %>% as.vector(),
    sigma = x[(2 * k + 1):(3 * k)] %>% as.vector(),
    iter = iter,
    grad = grad_list,
    u_frame = u_frame,
    sigma_frame = sigma_frame,
    pi_frame = pi_frame
  )
  
  return(result_list)
}

