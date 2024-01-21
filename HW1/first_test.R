# k=2 case

library(dplyr)

u1 = 5
sigma1 = 2

u2=10
sigma2=1/3

pi_1 = 1/3
pi_2 = 2/3

n=1000

uniform_random = runif(n,0,1)

data_points = (uniform_random>pi_1) * rnorm(n, u1, sigma1) + (uniform_random<=pi_1) * rnorm(n, u2, sigma2)


u_init = c(1,10)
sigma_init = c(2,5)
pi_init = c(1,3)


density_list = list()
for(i in 1:k){
  density_list[[i]] = dnorm(data_points,u[i],sigma[i])
}
density_matrix=data.frame(density_list) %>% as.matrix()



#dnorm(data_points,u_init[1],sigma_init[1])

# loss function

loss_function = function(data_points,u,sigma,pi_value){
  n = length(data_points)
  k = length(u)
  loss = 0
  
  pi_value = pi_value^2/sum(pi_value^2)
  
  density_list = list()
  for(i in 1:k){
    density_list[[i]] = dnorm(data_points,u[i],sigma[i])
  }
  density_matrix=data.frame(density_list) %>% as.matrix()
  
  loss = density_matrix %*% pi_value %>% log() %>% sum()
  
  return(-loss)
}

#loss_function(data_points,u_init,sigma_init,pi_init)





# gradient function for u

pi_grad = function(density_matrix, u, sigma, pi_value) {
  
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
        second_term_matrix[i, j] = -2 * pi_value[i] / (sum(pi_value^2)^2)
      }
    }
  }
  
  pi_grad_vec = ((density_matrix %*% second_term_matrix) * inverse_term)  %>% colSums()
  
  return(-pi_grad_vec)
  
}

grad_calculate = function(data_points,u,sigma,pi_value){
  
  density_list = list()
  k = length(u)
  for (i in 1:k)
  {
    density_list[[i]] = dnorm(data_points, u[i], sigma[i])
  }
  density_matrix = data.frame(density_list) %>% as.matrix()
  
  pi_grad_vec = pi_grad(density_matrix, u, sigma, pi_value)
  
}


centered_data_point = list()

for(i in 1:k)
{
  centered_data_point[[i]] = (data_points - u[i]) / sigma[i]
}

centered_data_point_matrix = data.frame(centered_data_point) %>% as.matrix()


pi_square = (pi_value^2 / sum(pi_value^2) ) 

phi_pi =  density_matrix  * pi_square

u_grad = (phi_pi * centered_data_point_matrix) %>% colSums() %>% as.vector()










