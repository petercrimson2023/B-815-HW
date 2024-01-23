library(mclust)



set.seed(123)

u1 = 1
sigma1 = 1/3

u2=-1
sigma2=1/5

pi_1 = 1/3
pi_2 = 2/3

n=10000


uniform_random = runif(n,0,1)

data_points = (uniform_random<=pi_1) * rnorm(n, u1, sigma1) + (uniform_random>pi_1) * rnorm(n, u2, sigma2)





gmm_model = Mclust(data_points)


gmm_model$parameters


plot(gmm_model, data_points)





