################################################################################
#
#                                                                                     
#   Filename    :    Simulate survival datasets .R    												  
#   Project     :    Article "Buckley-James Boosting Model based on Extreme Learning Machine and Random Survival Forests" 
#   Authors     :    Jianfen Kong, Shuhong Zhang                                                              
#   Date        :    20/02/2023
#   Purpose     :    File to store the support functions to simulate survival datasets for simulation studies.
#   Input data files  :    ---                                                        
#   Output data files :    ---
#   R Version   :    R-4.1.3                                                              
#   Required R packages :  missForest_1.4,
#                          MASS_7.3-55,
#                          compositions_2.0-4,
#                          survival_3.2-13
#                          purrr_1.4
################################################################################

library(survival)
library(compositions)

# 1. Covariates without missing ------------------------------------------------

#(1)Linear without missing.
pro_surv_linear = function(n){
  
  #The function is used to simulate linear effects survival dataset based on the setting of linear effects data 
  #in our experiments.
  
  #ARGUMENTS
  #n: Number of samples in the simulated survival dataset.
  
  #RETURN
  #data: A dataframe with linear effects survival dataset.
  #censoring_rate: the censoring rate of the simulated survival dataset.
  
  #step1: Produce covariates X.
  b = rep(0.4, 15)
  p = 30
  mu = rep(0, p)
  J = matrix(1, ncol = 5, nrow = 5)
  diag(J) = 1.01
  sigma = matrix(0, ncol = p, nrow = p)
  sigma[1:5, 1:5] = sigma[6:10, 6:10] = sigma[11:15,11:15] = J
  diag(sigma)[16:p] = 1
  x = mvrnorm(n = n, mu = mu,Sigma = sigma)
  
  #step2: Produce survival time T.
  beta = rep(0, p)
  beta[1:length(b)] = b
  epsilon = rnorm(n,0,3)
  Sur = exp(0.5 + x%*%beta + epsilon)
  
  #step3: Produce censoring time C.
  CT = runif(n, 0, 20)   
  
  #step4: Obtain the observation of response variable.
  surv_time = matrix(pmin(Sur, CT), ncol = 1)
  status = matrix(as.numeric(Sur<=CT),ncol = 1)
  censoring_rate = 1-(sum(status)/n)
  
  data = cbind(surv_time, status, x)
  data_df = as.data.frame(data,col.names = 1:ncol(data))
  colnames(data_df) = c('time', 'status', paste('x', 1:p, sep = ''))
  out = list('data' = data_df, 'censoring_rate' = censoring_rate)
  return(out)
}

#(2)Nonlinear without missing.
pro_surv_nonlinear = function(n){
  
  #The function is used to simulate non-linear effects survival dataset based on the setting of non-linear effects data 
  #in our experiments.
  
  #ARGUMENTS
  #n: Number of samples in the simulated survival dataset.
  
  #RETURN
  #data: A dataframe with non-linear effects survival dataset.
  #censoring_rate: The censoring rate of the simulated survival dataset.
  
  #step1: Produce covariates X.
  p = 4
  mu = rep(0,p)
  sigma = matrix(unlist(map(1:p, function(i) map_dbl(1:p, function(j) 0.7^abs(i-j)))), ncol = p)
  x  = mvrnorm(n = n,mu = mu, Sigma = sigma)
  colnames(x) = paste('x', 1:p, sep = '')
  
  #step2: Produce non-linear survival time T.
  f_x = 4*x[,1]^2 + x[, 1] + sin(6*x[,2]) + cos(6*x[ , 3]-1) + 4*x[ , 4]^3 + x[ , 4]^2
  epsilon = rnorm(n, 0, 0.75)
  Sur = exp(f_x + epsilon)
  
  #step3: Produce time C.
  CT = exp(rnorm(n, 0, 0.75))   
  
  #step4: Obtain the observation of response variable.
  surv_time = matrix(pmin(Sur, CT), ncol = 1)
  status = matrix(as.numeric(Sur <= CT), ncol = 1)
  censoring_rate = 1 - (sum(status)/n)
  
  data = cbind(surv_time, status, x)
  data_df = as.data.frame(data, col.names = 1:ncol(data))
  colnames(data_df) = c('time', 'status', paste('x', 1:p, sep = ''))
  out = list('data' = data_df, 'censoring_rate' = censoring_rate)
  return(out)
}



# 2. Covariates with missing ---------------------------------------------------

#(1)linear with missing.
pro_surv_linear_miss = function(n, NA_rate){
  
  #The function is used to simulate linear effects survival dataset with missing values on covariates.
  
  #ARGUMENTS
  #n: Number of samples in the simulated survival dataset. In our study,we determine it 0.1
  #NA_rate: A numeric value ranging 0 to 1 to specify the missing rate of covariates. In our study,
  #we determine it 0.1.
  
  #RETURN
  #data: A dataframe with linear effects survival dataset which has 20% missing values on covariates.
  #censoring_rate: The censoring rate of the simulated survival dataset
  
  #step1: Produce covariates X.
  p = 30  #the number of covariates is 30
  b = rep(0.4, 15)
  mu = rep(0, p)
  J = matrix(1,ncol = 5,nrow = 5)
  diag(J) = 1.01
  sigma = matrix(0, ncol = p, nrow = p)
  sigma[1:5, 1:5] = sigma[6:10, 6:10] = sigma[11:15, 11:15] = J
  diag(sigma)[16:p] = 1
  x = mvrnorm(n = n, mu = mu, Sigma = sigma)
  
  #step2: Produce survival time T.
  beta = rep(0,p)
  beta[1:length(b)] = b
  epsilon = rnorm(n, 0, 3)
  Sur = exp(0.5 + x%*%beta + epsilon)
  
  #step3: Produce censoring time C.
  CT = runif(n, 0, 20)    #The censoring time is generated from an uniform distribution [0,20].
  
  #step4: Obtain the observation of response variable.
  surv_time = matrix(pmin(Sur, CT), ncol = 1)
  status = matrix(as.numeric(Sur <= CT), ncol = 1)
  censoring_rate = 1 - (sum(status)/n)
  
  #step5: Delete 20% data of covariates based on the assumption of missing at random (MAR).
  miss_ind = sample(1:n, round(1*n))
  x_miss = simulateMissings(x = as.data.frame(x[miss_ind, ]), 
                            MARprob = NA_rate)
  x1 = rbind(x_miss, x[-miss_ind, ])
  surv_time1 = c(surv_time[miss_ind, ], surv_time[-miss_ind, ])
  status1 = c(status[miss_ind, ], status[-miss_ind, ])
  
  data = cbind(surv_time1, status1, x1)
  data1 = data[sample(1:n, n), ]
  data_df = as.data.frame(data1)
  colnames(data_df) = c('time', 'status', paste('x', 1:p, sep = ''))
  out = list('data' = data_df, 'censoring_rate' = censoring_rate)
  return(out)
}

#(2)non-linear with missing.

pro_surv_nonlinear_miss = function(n, NA_rate){
  
  #The function is used to simulate non-linear effects survival dataset with missing values on covariates.
  
  #ARGUMENTS
  #n: Number of samples in the simulated survival dataset. In our study,we determine it 0.1
  #NA_rate: A numeric value ranging 0 to 1 to specify the missing rate of covariates. In our study,
  #we determine it 0.1.
  
  #RETURN
  #data: A dataframe with nonlinear effects survival dataset which has 20% missing values on covariates.
  #censoring_rate: The censoring rate of the simulated survival dataset
  
  
  #step1: Produce covariates X.
  p = 4
  mu = rep(0, p)
  sigma = matrix(unlist(map(1:p, function(i) map_dbl(1:p, function(j) 0.7^abs(i-j)))), ncol = p)
  x = mvrnorm(n = n, mu = mu, Sigma = sigma)
  colnames(x) = paste('x', 1:p, sep = '')
  
  #step2: Produce non-linear survival time T.
  f_x = 4*x[, 1]^2 + x[, 1] + sin(6*x[, 2]) + cos(6*x[, 3] - 1) + 4*x[, 4]^3 + x[, 4]^2
  epsilon = rnorm(n, 0, 0.75)
  Sur = exp(f_x + epsilon)
  
  #step3: Produce censoring time C.
  CT = exp(rnorm(n, 0, 0.75))   
  
  #step4: Obtain the observation of response variable.
  surv_time = matrix(pmin(Sur, CT), ncol = 1)
  status = matrix(as.numeric(Sur <= CT), ncol = 1)
  y = Surv(time = surv_time, event = status, type = 'right')
  censoring_rate = 1-(sum(status)/n)
  
  #step5: Delete 20% data of covariates based on the assumption of missing at random (MAR)
  miss_ind = sample(1:n, round(1*n))
  x_miss = simulateMissings(x = as.data.frame(x[miss_ind, ]), 
                            MARprob = NA_rate)
  x1 = rbind(x_miss, x[-miss_ind, ])
  surv_time1 = c(surv_time[miss_ind, ], surv_time[-miss_ind, ])
  status1 = c(status[miss_ind, ], status[-miss_ind, ])
  
  data = cbind(surv_time1, status1, x1)
  data1 = data[sample(1:n, n), ]
  data_df = as.data.frame(data1)
  colnames(data_df) = c('time', 'status', paste('x', 1:p, sep = ''))
  out = list('data' = data_df, 'censoring_rate' = censoring_rate)
  return(out)
}

