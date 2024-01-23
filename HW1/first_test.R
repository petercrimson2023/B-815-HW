# k=2 case

library(dplyr)

u1 = 5
sigma1 = 2

u2=20
sigma2=8

pi_1 = 1/3
pi_2 = 2/3

n=1000

uniform_random = runif(n,0,1)

data_points = (uniform_random>pi_1) * rnorm(n, u1, sigma1) + (uniform_random<=pi_1) * rnorm(n, u2, sigma2)

#data_points = pi_1 * rnorm(n, u1, sigma1) + pi_2 * rnorm(n, u2, sigma2)

data_points %>% density() %>% plot()

data_points %>% hist()

# testing normality statistically

# Shapiro-Wilk test

#shapiro.test(data_points)


u_init = c(1,2)
sigma_init = c(2,5)
pi_init = c(1,3)

u= u_init
sigma = sigma_init
pi_value = pi_init


x = c(pi_value,u,sigma)

# loss function

loss_function = function(x,data_points=data_points){
  n = length(data_points)
  k = length(x)/3
  loss = 0
  
  pi_value = x[1:k]
  
  density_list = list()
  for(i in 1:k){
    density_list[[i]] = dnorm(data_points,x[k+i],x[2*k+i])
  }
  density_matrix=data.frame(density_list) %>% as.matrix()
  
  loss = density_matrix %*% pi_value %>% log() %>% sum()
  
  return(-loss)
}



#View(density_matrix)



# gradient function for u

pi_grad = function(density_matrix, u, sigma, pi_value) {
  
  k = length(u)
  
  inverse_term = (density_matrix %*% pi_init^2/sum(pi_init^2)) %>% as.vector()
  
  inverse_term = 1 / inverse_term
  
  second_term_matrix =  matrix(0, nrow = k, ncol = k)
  
  for (i in 1:k)
  {
    for (j in 1:k)
    {
      if (i == j)
      {
        second_term_matrix[i, j] = 2 * pi_value[i] * sum(pi_value[-i]^2) / (sum(pi_value^2)^2)
      }
      else
      {
        second_term_matrix[i, j] = -2 * pi_value[j] / (sum(pi_value^2)^2)
      }
    }
  }
  
  pi_grad_vec = ((density_matrix %*% second_term_matrix) * inverse_term)  %>% colSums()
  
  return(-pi_grad_vec)
  
}

u_grad = function(centered_data_point_matrix, density_matrix, pi_value)
{
  pi_square = (pi_value^2 / sum(pi_value^2) ) 
  
  phi_pi =  density_matrix  * pi_square
  
  u_grad = (phi_pi * centered_data_point_matrix) %>% colSums() %>% as.vector()
  
  return(-u_grad)
}


sigma_grade = function(third_grad_centered_matrix,density_matrix, pi_value)
{
  pi_square = (pi_value^2 / sum(pi_value^2) )
  
  sigma_grad = (density_matrix * pi_square * third_grad_centered_matrix) %>% colSums() %>% as.vector()
  
  return(-sigma_grad)
}


grad_calculate = function(data_points, u, sigma, pi_value)
{
  density_list = list()
  centered_data_point = list()
  third_grad_centered = list()
  
  
  k = length(u)
  
  for (i in 1:k)
  {
    density_list[[i]] = dnorm(data_points, u[i], sigma[i])
    
    centered_data_point[[i]] = (data_points - u[i]) / sigma[i]
    
    third_grad_centered[[i]] = (data_points - u[i])^2 / (sigma[i]^3) - 1 / sigma[i]
    
  }
  
  #used in calculating all of the gradients
  density_matrix = data.frame(density_list) %>% as.matrix()
  
  #used in calculating the gradient of u
  centered_data_point_matrix = data.frame(centered_data_point) %>% as.matrix()
  
  #used in calculating the gradient of sigma
  third_grad_centered_matrix = data.frame(third_grad_centered) %>% as.matrix()
  
  #calculating the gradient
  
  pi_grad_vec = pi_grad(density_matrix, u, sigma, pi_value)
  u_grad_vec = u_grad(centered_data_point_matrix, density_matrix, pi_value)
  sigma_grad_vec = sigma_grade(third_grad_centered_matrix, density_matrix, pi_value)
  
  #combine the gradients
  
  grad_vec = c(pi_grad_vec, u_grad_vec, sigma_grad_vec)
  
  return(grad_vec)
  
}



# gradient descent

parameter_init = c(pi_init,u_init,sigma_init)

grad_descent = function(data_points, parameter_init, step_size, max_iter,eps = 1e-5)
{
  parameter = parameter_init
  parameter_old = rep(0, length(parameter))
  
  k = length(parameter_init) / 3
  
  for (i in 1:max_iter)
  {
    pi_value = parameter[1:k]
    u = parameter[(k+1):(2*k)]
    sigma = parameter[(2*k+1):(3*k)]
    
    grad_vec = grad_calculate(data_points, u, sigma, pi_value)
    
    parameter_old = parameter
    parameter = parameter - step_size * grad_vec
    
    loss = loss_function(data_points,u, sigma, pi_value)
    
    #i th iteration result:
    
    text = paste0("iteration: ",i," Loss is: ", loss)
    print(text)
    print(as.vector(parameter))
    print("\n")
    
    if (max(abs(parameter - parameter_old)) < eps)
    {
      break
    }
    
  }
  
  #return the final result
  
  return(parameter)
  
}


grad_descent(data_points,parameter_init,0.05,1000)







