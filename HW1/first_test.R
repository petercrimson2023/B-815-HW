# k=2 case

u1 = 5
sigma1 = 2

u2=10
sigma2=1/3

pi_1 = 1/3
pi_2 = 2/3

n=1000

uniform_random = runif(n,0,1)

data_points = (uniform_random>pi_1) * rnorm(n, u1, sigma1) + (uniform_random<=pi_1) * rnorm(n, u2, sigma2)


u_init = c(1,1)
sigma_init = c(2,2)
pi_init = c(0.5,0.5)



dnorm(data_points,u_init[1],sigma_init[1])

# loss function

loss_function = function(data_points,u,sigma,pi_value){
  n = length(data_points)
  k = length(u)
  loss = 0
  density_list = list()
  for(i in 1:k){
    density_list[[i]] = dnorm(data_points,u[i],sigma[i])
  }
  density_matrix=data.frame(density_list) %>% as.matrix()
  
  loss = density_matrix %*% pi_value %>% log() %>% sum()
  
  return(-loss)
}


