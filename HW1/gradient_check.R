# gradient testing


# Gradient check


grad_function(x,data_points) %>% as.vector()

numerical_grad_list = c()

for(i in 1:length(x)){
  x_epsilon = x
  x_m_epsilon = x
  x_epsilon[i] = x[i] + 1e-5
  x_m_epsilon[i] = x[i] - 1e-5
  value = ((loss_function(x_epsilon,data_points) - loss_function(x_m_epsilon,data_points))/(2*1e-5)) %>% as.vector()
  numerical_grad_list = c(numerical_grad_list,value)
}

numerical_grad_list

#Checking grad for squared version

x = c(5,4,0,0,0.5,0.2)


numerical_grad_list = c()

for(i in 1:length(x)){
  x_epsilon = x
  x_m_epsilon = x
  x_epsilon[i] = x[i] + 1e-6
  x_m_epsilon[i] = x[i] - 1e-6
  value = ((loss_function(x_epsilon,data_points) - loss_function(x_m_epsilon,data_points))/(2*1e-6)) %>% as.vector()
  numerical_grad_list = c(numerical_grad_list,value)
}

numerical_grad_list
grad_function_absolute(x,data_points) %>% as.vector()


