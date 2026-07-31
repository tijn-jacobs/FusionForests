################################################################################
#
#                                                                                     
#   Filename    :    BJ_ELM.R    												  
#   Project     :    Article "Buckley-James Boosting Model based on Extreme
#                    Learning Machine and Random Survival Forests"                                                             
#   Authors     :    Jianfen Kong, Shuhong Zhang                                                              
#   Date        :    20/02/2023
#   Purpose     :    Functions allowing to fit ELM-based BJ boosting model, obtain predicted values using a BJ-ELM model 
#                    and take k-fold cross-validation for BJ-ELM model.
#   Input data files  :    ---                                                        
#   Output data files :    ---
#
#   R Version   :    R-4.1.3                                                                
################################################################################


source('BJ_ELM_Code/R/Main_Functions/ELM_Boosting.R')


#1. Model training------------------------------------

cumsumsurv = function (x){
  #Function to compute jumps of Kapla-Meier given Lagrange multiplier lambda.
  
  if (any(is.na(x))) 
    stop("NaNs")
  s = x
  .C("cumsumsurv", x = as.numeric(x), s = as.numeric(s), LLL = length(x))$s
}

iter = function(x, y, status, y_hat, nhid, nbase){
  
  # Function to take iterations of the BJ-ELM model. 
  
  #ARGUMENTS
  #x: A dataframe with data of covariates.
  #y: A vector of the logarithm of survival time.
  #status: A vector of censoring indicator, must be 0 or 1 with 0=alive, 1=dead.
  #nihd: The number of hidden neurons. Defaults to 20. Generally, choosing it at the range of [5,40] is proper.
  #nbase: The number of base learners Mstop of boosting. Defaults to 25. Generally, choosing it at the range of [10,40] is proper.
  #y_hat: Estimated y from the training sample at previous iteration.
  
  #RETURN
  #Estimated y from the training sample at current iteration.
  
  N = length(status)
  u = y_hat
  res = y - u
  niceorder = order(res, - status) 
  resorder = res[niceorder]     
  dorder = status[niceorder] 
  dorder[N] = 1              
  uorder = u[niceorder] 
  ystar = y[niceorder]  
  xorder = as.matrix(x[niceorder, ])
  
  temp = WKM(x = resorder, d = dorder, zc = 1:N) 
  
  jifen = cumsumsurv(resorder * temp$jump)  
  Sresorder = temp$surv
  for (i in 1:N) if (dorder[i] == 0) {
    ystar[i] = uorder[i] + jifen[i]/Sresorder[i]
  }
  
  model = ELM_boosting(x = xorder, y = ystar, nhid = nhid, nbase = nbase, step = 0.1)
  return(model)
}


BJ_ELM = function (x, y, status,nhid = 20, nbase = 25) {
  # Function for training the BJ-ELM model. 
  
  #ARGUMENTS
  #x: A dataframe with data of covariates.
  #y: A vector of the logarithm of survival time.
  #status: A vector of censoring indicator, must be 0 or 1 with 0=alive, 1=dead.
  #nihd: The number of hidden neurons. Defaults to 20. Generally, choosing it at the range of [5,40] is proper.
  #nbase: The number of base learners Mstop of boosting. Defaults to 25. Generally, choosing it at the range of [10,40] is proper.
  
  #RETURN
  #y_hat: Estimated y from the training sampleS. 
  
  maxiter = 30
  error = 1e-05
  x = as.matrix(x)
  model = vector('list', length = 3)
  model[[1]] = ELM_boosting(x = x, y = y,nhid = nhid, nbase = nbase, step = 0.1)
  f_x_hat = matrix(NA, ncol = nrow(x), nrow = 3)
  f_x_hat[1, ] = model[[1]]$y_hat 
  for (i in 2:3) {
    model[[i]] = iter(x = x,y = y, status = status, y_hat = f_x_hat[i-1, ],
                      nhid = nhid,nbase = nbase)
    f_x_hat[i, ] = model[[i]]$y_hat
  }
  k = 2
  while (k <= maxiter && error <= sum(abs(f_x_hat[2, ] - 
                                          f_x_hat[3, ]))) {
    f_x_hat[2, ] = f_x_hat[3, ]
    model[[2]] = model[[3]]
    model[[3]] = iter(x = x, y = y, status = status, y_hat = f_x_hat[2, ], nhid = nhid, nbase = nbase)
    f_x_hat[3, ] = model[[3]]$y_hat
    k = k + 1
  }
  
  list(model = model[[3]])
}


#2. Prediction --------------------------------------------------------------

pre_BJ_ELM = function(model, newx){
  #Function to obtain predicted values using a BJ-ELM model.
  
  #ARGUMENTS
  #model: A BJ-ELM model fitted with function 'BJ_ELM()'.
  #newx: A dataframe of the covariates of testing data.
  
  #RETURN
  #y_pre: Estimated the logarithm of survival time from the testing samples.
  
  hid_weights = model$hid_weights
  bias = model$bias
  beta = model$output_weights
  outweight = model$outweight
  a = model$a
  step = model$step
  y_pre_row = map_dfc(1:ncol(beta), 
                      function(x) actfun(newx%*%as.matrix(hid_weights[[x]]) +
                                           matrix(rep(bias[, x], nrow(newx)), nrow = nrow(newx), byrow = TRUE))%*%matrix(beta[, x], ncol = 1))
  y_pre_row = as.matrix(y_pre_row)
  y_pre = step*apply(y_pre_row, 1, sum)
  return(y_pre)
}


# 3. K-folds cross validation -----------------------------------------------------

CV_BJ_ELM = function(data, k_folds = 5, nbase = seq(10, 35, 5), nhid = seq(5, 30, 5), seed = 1){
  
  #Function to take k-fold cross-validation for BJ-ELM model.
  
  #ARGUMENTS
  #data: A dataframe of training data.
  #k_folds: Number of fold of cv. Default to 5.
  #nbase: Optional user-supplied Mstop sequence. Default to (10, 15, 20, 25, 30, 35).
  #nhid: Optional user-supplied sequence of the number of hidden neurons. Default to (5, 10, 15, 20, 25, 30).
  #seed: An integer, random seed. Default to 1.
  #newx: A dataframe the covariates of testing data.
  
  #RETURN
  #best_nbase: The best Mstop chosen by k-fold CV.
  #best_nhid: The best nhid chosen by k-fold CV.
  
  data = data[, c(which(colnames(data) == 'time'), which(colnames(data) == 'status'),
                  which(colnames(data) != 'status' & colnames(data) != 'time'))]
  tmp1 = expand.grid(nbase, nhid)
  tmp1 = cbind( tmp1[, 1], tmp1[, 2])
  nn1 = dim(tmp1)[1]
  C_index = matrix(NA, nrow=nn1, ncol=k_folds)
  set.seed(seed = seed)
  folds = createFolds(y = data$time, k = k_folds)
  
  for(k in 1:k_folds){
    test = data[folds[[k]], ]
    train = data[-folds[[k]], ]
    for (i in 1:nn1) {
      BJ_ELM = BJ_ELM(x = train[ , -(1:2)], y = log(train$time), status = train$status,
                      nhid = tmp1[i, 2], nbase = tmp1[i, 1])
      y_pre = pre_BJ_ELM(model = BJ_ELM$model, newx = as.matrix(test[, -(1:2)]))
      C_index[i,k ] = estC(timeX = test$time, statusX = test$status, scoreY = exp(y_pre))
    }
  }
  C_index_out = apply(C_index, 1, mean)
  max_ind = which.max(C_index_out)
  best_nbase = tmp1[max_ind, 1]
  best_nhid = tmp1[max_ind, 2]
  out = list('best_nbase' = best_nbase, 'best_nhid' = best_nhid)
  return(out)
}


