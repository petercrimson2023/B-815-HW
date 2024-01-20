#Problem Discription

#1 Derive and implement the gradient descent algorithm for the univariate Gaussian mixture model (GMM) estimation problem. 
#  hint: need to take special care about the weight parameters with the unit-sum constraint; consider 
#  1) reparameterization, 2) approximated treatment 

#2.Implement the BB method and accelerated gradient method for the GMM

# Prpblem 1

# case k=2

u1 = 1
sigma1 = 2

u2=10
sigma2=1/3

pi_1 = 1/3
pi_2 = 2/3

n=1000

uniform_random = runif(n,0,1)

data_points = (uniform_random>pi_1) * rnorm(n, u1, sigma1) + (uniform_random<=pi_1) * rnorm(n, u2, sigma2)

# plotting hit

hist(data_points,breaks=50,main="Histogram of data points",xlab="data points",ylab="frequency")


# plotting kde

plot(density(data_points),main="KDE of data points",xlab="data points",ylab="density")


# case k=5

u<-c(1,10,20,30,100)

sigma<-c(1,2,3,4,10)

pi<-c(1/3,1/5,1/10,13/60,0.15)

cum_sum <- cumsum(pi)

n=1000

uniform_random = runif(n,0,1)

data_matrix = cbind(rnorm(n, u[1], sigma[1]),rnorm(n, u[2], sigma[2]),rnorm(n, u[3], sigma[3]),rnorm(n, u[4], sigma[4]),rnorm(n, u[5], sigma[5]))

data_points = data_matrix[,sapply(uniform_random,function(x){sum(x>cum_sum)+1})]

# plotting hit

hist(data_points,breaks=50,main="Histogram of data points",xlab="data points",ylab="frequency")

# plotting kde

plot(density(data_points),main="KDE of data points",xlab="data points",ylab="density")


