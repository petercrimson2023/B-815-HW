#Nestrov Accleration


default_data_generation = function(n=100) {
  set.seed(123)
  
  u1 = 1
  sigma1 = 0.1
  
  u2=-5
  sigma2=1
  
  pi_1 = 1/3
  pi_2 = 2/3
  
  uniform_random = runif(n,0,1)
  
  data_points = (uniform_random<=pi_1) * rnorm(n, u1, sigma1) + (uniform_random>pi_1) * rnorm(n, u2, sigma2)
  
  data_points %>% density() %>% plot()
  
  #data_points %>% hist(,breaks=50)
  
  return(data_points)
}


data_points = default_data_generation(n=1000)



loss_function_square = function(x, data_points = data_points) {
  n = length(data_points)
  k = length(x) / 3
  loss = 0
  
  max_gamma = max(x[1:k])
  
  pi_value = exp(x[1:k]-max_gamma) / sum(exp(x[1:k]-max_gamma))
  
  density_list = list()
  for (i in 1:k) {
    density_list[[i]] = dnorm(data_points, x[k + i], x[2 * k + i]^2)
  }
  density_matrix = data.frame(density_list) %>% as.matrix()
  
  loss = (density_matrix %*% pi_value) + 1e-5
  
  loss = loss %>% log() %>% sum()
  
  return(-loss)
}

#optim(x,loss_function_square,data_points = data_points,method="BFGS",control=list(trace=1))

grad_function_square = function(x, data_points)
{
  n = length(data_points)
  k = length(x) / 3
  loss = 0
  
  max_gamma = max(x[1:k])
  
  pi_value = exp(x[1:k]-max_gamma) / sum(exp(x[1:k]-max_gamma))
  
  exp_gamma_density_list = list() # exp(gamma_i) * density_i matrix used in claculating all gradients
  
  centered_data_point_list = list() # ( - u_i) / sigma_i^2 matrix used in calculating gradient of u
  
  second_order_centered_data_point_list = list() # (x_ - u_i)^2 / sigma_i^5 matrix used in calculating gradient of sigma
  
  sum_exp_gamma_density = rep(0, n)
  
  for (i in 1:k) {
    
    exp_gamma_density_value = dnorm(data_points, x[k + i], x[2 * k + i]^2) * exp(x[i]-max_gamma)
    
    exp_gamma_density_list[[i]] =
      exp_gamma_density_value
    
    sum_exp_gamma_density =
      sum_exp_gamma_density + exp_gamma_density_value + 1e-8
    
    centered_data_point_list[[i]] =
      (data_points - x[k + i]) / (x[2 * k + i]^4)
    
    second_order_centered_data_point_list[[i]] =
      2* (data_points - x[k + i])^2 / (x[2 * k + i] ^ 5) - 2 / (x[2 * k + i])
    
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




grad_function_absolute = function(x, data_points)
{
  n = length(data_points)
  k = length(x) / 3
  loss = 0
  
  max_gamma = max(x[1:k])
  
  pi_value = exp(x[1:k]-max_gamma) / sum(exp(x[1:k]-max_gamma))
  
  exp_gamma_density_list = list() # exp(gamma_i) * density_i matrix used in claculating all gradients
  
  centered_data_point_list = list() # ( - u_i) / sigma_i^2 matrix used in calculating gradient of u
  
  second_order_centered_data_point_list = list() # (x_ - u_i)^2 / sigma_i^5 matrix used in calculating gradient of sigma
  
  sum_exp_gamma_density = rep(0, n)
  
  for (i in 1:k) {
    
    exp_gamma_density_value = dnorm(data_points, x[k + i], abs(x[2 * k + i])) * exp(x[i]-max_gamma)
    
    exp_gamma_density_list[[i]] =
      exp_gamma_density_value
    
    sum_exp_gamma_density =
      sum_exp_gamma_density + exp_gamma_density_value + 1e-8
    
    centered_data_point_list[[i]] =
      (data_points - x[k + i]) / (x[2 * k + i]^4)
    
    second_order_centered_data_point_list[[i]] =
      (data_points - x[k + i])^2 / (abs(x[2 * k + i]) ^ 3) *sign(x[2 * k + i]) - sign(x[2 * k + i]) / (x[2 * k + i])
    
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





gradient_Nestrov_descent = function(x,
                               data_points,
                               max_iter = 1000,
                               step_size = 0.01,
                               tol = 1e-3,
                               loss_tol=1e-5,
                               report = FALSE) {
  iter = 0

  k = length(x) / 3
  loss_list = c()
  grad_list = c()
  u_frame = data.frame()
  sigma_frame = data.frame()
  pi_frame = data.frame()
  
  x_old = x
  y_old = x
  
  loss_new = loss_function_square(x, data_points)
  
  if (is.nan(loss_new) || is.infinite(loss_new)) {
    stop("Initial loss is NaN or infinite")
  }
  
  loss_list = c(loss_list, loss_new)
  
  u_frame=rbind(x[(k+1):(2*k)],u_frame)
  sigma_frame=rbind(x[(2*k+1):(3*k)],sigma_frame)
  pi_frame=rbind(exp(x[1:k])/sum(exp(x[1:k])),pi_frame)
  
  lambda_old = 0.01

  
  while (iter < max_iter) {
    
    iter = iter + 1
    
    gradient = grad_function_absolute(y_old, data_points)
    
    x = y_old - step_size * (1+1/(0.01*iter)) * gradient
    
    

    # Checking if gradient is NaN or infinite
    if (any(is.nan(gradient)) || any(is.infinite(gradient))) {
      
      print(as.vector(y_old))
      names(pi_frame) = paste0("pi_",1:k)
      names(u_frame) = paste0("u_",1:k)
      names(sigma_frame) = paste0("sigma_",1:k)
      
      result_list = list(
        loss = loss_list,
        pi_value = (exp(x[1:k]) / sum(exp(x[1:k]))) %>% as.vector() ,
        u = x[(k + 1):(2 * k)] %>% as.vector(),
        sigma = (x[(2 * k + 1):(3 * k)]^2) %>% as.vector(),
        iter = iter,
        grad = grad_list,
        u_frame = u_frame,
        sigma_frame = sigma_frame,
        pi_frame = pi_frame
      )
      
      print(paste0("Gradient is NaN or infinite at iteration ", iter))
      
      return(result_list)
    }
  
    lambda = (1 + sqrt(1 + 4 * lambda_old^2)) / 2
    
    #y_new = x + iter / (iter + 3) * (x - x_old)
    
    y_new = x + (lambda_old - 1) / lambda * (x - x_old)
    
    u_frame=rbind(x[(k+1):(2*k)],u_frame)
    sigma_frame=rbind(x[(2*k+1):(3*k)]^2,sigma_frame)
    pi_frame=rbind(exp(x[1:k])/sum(exp(x[1:k])),pi_frame)
 
    loss_old = loss_new
    loss_new = loss_function_square(x, data_points)
    if (is.nan(loss_new) || is.infinite(loss_new)) {
      
      names(pi_frame) = paste0("pi_",1:k)
      names(u_frame) = paste0("u_",1:k)
      names(sigma_frame) = paste0("sigma_",1:k)
      
      result_list = list(
        loss = loss_list,
        pi_value = (exp(x[1:k]) / sum(exp(x[1:k]))) %>% as.vector() ,
        u = x[(k + 1):2 * k] %>% as.vector(),
        sigma = (x[(2 * k + 1):(3 * k)]^2) %>% as.vector(),
        iter = iter,
        grad = grad_list,
        u_frame = u_frame,
        sigma_frame = sigma_frame,
        pi_frame = pi_frame
      )
      
      print(paste0("Loss is NaN or infinite at iteration ", iter))
      return(result_list)
    }
    
    loss_list = c(loss_list, loss_new)
    grad_list = c(grad_list, max(abs(gradient)))
    
    if (report) {
      cat("iter:", iter, "loss:", loss_new, "\n")
      cat("pi:", exp(x[1:k]) / sum(exp(x[1:k])), "\n")
      cat("u:", x[(k + 1):2 * k] , "\n")
      cat("sigma:", x[(2 * k + 1):(3 * k)]^2 , "\n")
    }
    
    # if (abs(loss_new - loss_old) < 1e-3) {
    #   break
    # }
    if (max(abs(x-x_old)) < tol ) {
      break
    }

    x_pld = x
    y_old = y_new
    
  }
  
  names(pi_frame) = paste0("pi_",1:k)
  names(u_frame) = paste0("u_",1:k)
  names(sigma_frame) = paste0("sigma_",1:k)
  
  result_list = list(
    loss = loss_list,
    pi_value = (exp(x[1:k]) / sum(exp(x[1:k]))) %>% as.vector() ,
    u = x[(k + 1):2 * k] %>% as.vector(),
    sigma = (x[(2 * k + 1):(3 * k)]^2) %>% as.vector(),
    iter = iter,
    grad = grad_list,
    u_frame = u_frame,
    sigma_frame = sigma_frame,
    pi_frame = pi_frame
  )
  
  return(result_list)
}



#_________________Test________________________



x = c(1,2,1,-5,sqrt(0.1),1) 

x = c(1,1+log(2),1,-5,sqrt(0.1),1 )#true point

#x = c(5,4,0,0,0.5,0.2) # another initial points 

result_list = gradient_Nestrov_descent(x,data_points,step_size = 1e-2/5,max_iter = 2000,report =FALSE)

result_list[2:5]
result_list$loss[1:result_list$iter] %>% plot( ,type="l",main="loss")
result_list$grad %>% plot( ,type="l",main="absolute norm of grad")
result_list$u_frame%>% plot(,main="u")
result_list$sigma_frame%>% plot(,main="sigma")
result_list$pi_frame%>% plot(,main="pi",ylim=c(0,1))

x = c(1,1+log(2),1,-5,sqrt(0.1),1 )
loss_values = c()
x_5 = seq(0.01,10,by=0.01)

for (i in 1:length(x_5)){
  x[5] = x_5[i]
  loss_values[i] = loss_function(x,data_points)
}

plot(x_5,loss_values,type="l",main="loss function with respect to sigma")

