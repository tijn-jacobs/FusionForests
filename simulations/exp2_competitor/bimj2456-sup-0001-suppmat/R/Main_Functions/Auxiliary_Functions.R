################################################################################
#
#                                                                                     
#   Filename    :    auxiliary functions.R    												  
#   Project     :    Article "Buckley-James Boosting Model based on Extreme
#                    Learning Machine and Random Survival Forests"                                                             
#   Authors     :    Jianfen Kong, Shuhong Zhang                                                              
#   Date        :    20/02/2023
#   Purpose     :    File to store the support functions for implementing the experiments of real data applications 
#                    and simulation studies.
#   Input data files  :    ---                                                        
#   Output data files :    ---
#   R Version   :    R-4.1.3                                                                
#   Required R packages :  
#                          purrr_0.3.4
#                          emplik_1.2
#                          Hmisc_4.6-0
#                          dplyr_1.0.10
#                          randomForestSRC_2.14.0
#                          pec_2021.10.11
#                          missForest_1.4
#                          SurvMetrics_0.5.0
#                          rms_6.2-0
################################################################################


library('Hmisc')
library('missForest')
library('pec')
library('SurvMetrics')
library('randomForestSRC')
library('compareC')
#1. Support functions for data preprocessing ----------------------------------------------------------

to_factor = function(x, threshold = 5){
  #An auxiliary function for real data applications, which converts a discrete variable into a factor or a numeric vector.
  
  # ARGUMENTS
  #x: A vector. 
  #threshold: A integer specifying the threshold. If x has less than 'threshold' different values, it will be converted 
  #           into a factor otherwise, it will be converted into a numeric vector. 
  
  #RETURN
  #A numeric vector or a factor.
  
  if(length(unique(x)) <= threshold){
    y = as.factor(x)
  }else{
    y = as.numeric(x)
  }
  return(y)
}

data_trans = function(data, threshold = 5){
  #An auxiliary function for real data applications, which converts each covariate of the real dataset into a factor or a numeric vector
  #based on the number of the different values of each covariates. For an ordered discrete variable, We convert it into a numeric variable; 
  #for a unordered discrete variable, we convert it into a factor; for a continuous variable, we keep it numeric.

  # ARGUMENTS
  #data: A dataframe with real survival data and covariates. The columns of survival time and censoring indicator
  #      should be named as 'time' and 'status' respectively.
  #threshold: A integer specifying the threshold. Each column having less than 'threshold' different values 
  #           will be converted into a factor, otherwise, it will be converted into a numeric vector. 
  
  #RETURN
  #A dataframe with real survival data and covariates.
  
  x = data[, which(colnames(data) !=  'status' & colnames(data) != 'time')]
  x1 = map_dfc(x, to_factor, threshold)
  data1 = as.data.frame(cbind(as.numeric(data[, which(colnames(data) == 'time')]), as.numeric(data[, which(colnames(data) == 'status')]), x1))
  colnames(data1) = c('time', 'status', colnames(x))
  return(data1)
}

z_score = function(train_x_con, test_x_con){
  
  #Function to standardize training data and testing data of the continuous variables.
  
  # ARGUMENTS
  #train_x_con: A dataframe, continuous covariates of training data.
  #test_x_con: A dataframe, continuous covariates of testing data.
  
  #RETURN
  #sta_train_x: A dataframe containing standardlized continuous covariates of training data.
  #sta_test_x: A dataframe containing standardlized continuous covariates of testing data.
  
  sta_train_x = as.data.frame(scale(as.matrix(train_x_con)))
  sta_test_x = imap_dfc(test_x_con, function(.x, .y) (.x-mean(train_x_con[[.y]]))/sd(train_x_con[[.y]]))
  colnames(sta_train_x) = colnames(sta_test_x) = colnames(train_x_con)
  out = list('sta_train_x' = sta_train_x, 'sta_test_x' = sta_test_x)
  return(out)
}  

onehot_encoding = function(train_x_dis, test_x_dis){
  
  #Function used to onehot encoding the discrete covariates of the training data and testing data.
  
  #ARGUMENTS
  #train_x_dis: A dataframe, discrete covariates of training data.
  #test_x_dis: A dataframe, discrete covariates of testing data.
  
  #RETURN
  #train_x_dis_onehot: A dataframe containing dummy variables of training data.
  #test_x_dis_onehot: A dataframe containing dummy variables of testing data.
  
  x_dis = as.data.frame(rbind(train_x_dis, test_x_dis))
  colnames(x_dis) = colnames(train_x_dis)
  formular1 = as.formula(paste(paste("~", 
                                     paste(colnames(x_dis), collapse = "+"))))
  x_dis_onehot = as.data.frame(model.matrix(formular1, x_dis))[, -1]
  train_x_dis_onehot = x_dis_onehot[1:nrow(train_x_dis), ]
  test_x_dis_onehot = x_dis_onehot[(nrow(train_x_dis)+1):nrow(x_dis), ]
  out = list('train_x_dis_onehot' = train_x_dis_onehot, 'test_x_dis_onehot' = test_x_dis_onehot)
  return(out)
}

impute_mean_mode = function(x){
  
  #Function to impute covariates with missing values. For discrete variables, it is used the mode of the variable to impute 
  #the missing values, while impute the missing values by the mean for continuous variable.
  
  # ARGUMENTS
  #x: A dataframe of covariates with missing values.
  
  #RETURN
  #A dataframe of imputed covariates.
  
  if(is.factor(x) == TRUE){
    object = unique(x)
    object1 = object[is.na(object) == FALSE]
    y = Hmisc::impute(x, object = object1)
  }else{
    y = Hmisc::impute(x, fun = mean)
  }
  return(y)
}

impute_RSF = function(train, test){
  
  #Function used to impute covariates with missing values by RSF.
  
  #ARGUMENTS
  #train: A dataframe with training data.
  #test: A dataframe with testing data.
  
  #RETURN
  #imp_train: A dataframe with imputed training dataset.
  #imp_test: A dataframe with imputed testing dataset.
  
  formula1 = as.formula(paste("Surv(time, status)~", paste(colnames(train[-(1:2)]), collapse = "+")))
  
  RSF = rfsrc(formula1, data  = train, na.action = "na.impute", 
              splitrule = "bs.gradient", nimpute = 5) 
  imp_train_x = RSF$xvar
  imp_train = as.data.frame(cbind(train[, 1:2], imp_train_x))
  colnames(imp_train) = colnames(train)
  
  predict1 = predict(object = RSF, newdata = test, na.action = "na.impute", nimpute  =  5)
  if(is.null(predict1$imputed.indv)){
    imp_test1 = test
  }else{
    imp_test_x = matrix(NA, ncol = ncol(test[, -(1:2)]), nrow = nrow(test))
    imp_test_x[predict1$imputed.indv, ] = as.matrix(predict1$imputed.data)[, -(1:2)]
    imp_test_x[setdiff(x = 1:nrow(test), y = predict1$imputed.indv), ] = as.matrix(test[setdiff(x = 1:nrow(test), y = predict1$imputed.indv), -(1:2)])
    imp_test = as.data.frame(cbind(as.matrix(test[, 1:2]), imp_test_x))
    imp_test1 = as.data.frame(apply(imp_test, 2, function(x)
      if(anyNA(as.numeric(x)) == TRUE){
        x
      }else{
        as.numeric(x)
      }))
    
  }
  colnames(imp_test1) = colnames(test)
  out = list('imp_train' = imp_train, 'imp_test' = imp_test1)
  return(out)
}

prepro_common = function(train, test, impute = 'no need', threshold = 5){
  
  #Function used to preprocess the training data and testing data, including the steps of imputing covariates with missing values
  #by 'impute_mean_mode()', standardlizing continuous covariates and converting the discrete covariates into dummy variables.
  
  #ARGUMENTS
  #train: A dataframe with training data.
  #test: A dataframe with testing data.
  #impute: Optional, imputation method for imputing covariates with missing values. If 'no need', it will not take imputation, 
  #        choosing 'mean_mode' will use the mode of the variable to impute the missing values for discrete variables, 
  #        while impute the missing values by the mean for continuous variable. Default to 'no need'. 
  #threshold: A integer specifying the threshold. Each column having less than 'threshold' different values 
  #           will be converted into a factor, otherwise, it will be converted into a numeric vector. 
  
  #RETURN
  #train_pro: A dataframe with preprocessed training dataset.
  #test_pro: A dataframe with preprocessed testing dataset.
  
  if(impute == 'no need'){  
    train1 = train
    test1 = test
  }else if(impute == 'mean_mode'){
    imp_train_x = map_dfc(train[, -(1:2)], impute_mean_mode)
    train_test_x = as.data.frame(rbind(test[, -(1:2)], imp_train_x))
    
    imp_train_test_x = map_dfc(train_test_x, impute_mean_mode)
    imp_test_x = imp_train_test_x[1:nrow(test), ]
    train1 = as.data.frame(cbind(train[, 1:2], imp_train_x))
    test1 = as.data.frame(cbind(test[, 1:2], imp_test_x))
    colnames(train1) = colnames(test1) = colnames(train)
    data1 = as.data.frame(rbind(train1, test1))
    data2 = data_trans(data1, threshold = threshold )
    train1 = data2[1:nrow(train1), ]
    test1 = data2[(nrow(train1)+1):nrow(data2), ]
  }
  
  train_x_con = select_if(train1[, which(colnames(train1) != 'status' & colnames(train1) != 'time')], is.numeric)
  test_x_con = select_if(test1[, which(colnames(test1) != 'status' & colnames(test1) != 'time')], is.numeric)
  z_score_standardlize = z_score(train_x_con = train_x_con, test_x_con = test_x_con)
  sta_train_x_con = z_score_standardlize$sta_train_x
  sta_test_x_con = z_score_standardlize$sta_test_x
  
  if(ncol(train_x_con) == ncol(train1[, -(1:2)])){
    train_pro = as.data.frame(cbind(train1[, which(colnames(train1) == 'time')], train1[, which(colnames(train1) == 'status')], 
                                    sta_train_x_con))
    test_pro = as.data.frame(cbind(test1[, which(colnames(test1) == 'time')], test1[, which(colnames(test1) == 'status')], 
                                   sta_test_x_con))
  }else{
    train_x_dis = select_if(train1[, which(colnames(train1) != 'status' & colnames(train1) != 'time')], is.factor)
    test_x_dis = select_if(test1[, which(colnames(test1) != 'status' & colnames(test1) != 'time')], is.factor)
    onehot = onehot_encoding(train_x_dis = train_x_dis, test_x_dis = test_x_dis)
    oh_train_x_dis = onehot$train_x_dis_onehot
    oh_test_x_dis = onehot$test_x_dis_onehot
    train_pro = as.data.frame(cbind(train1[, which(colnames(train1) == 'time')], train1[, which(colnames(train1) == 'status')], 
                                    oh_train_x_dis, sta_train_x_con))
    test_pro = as.data.frame(cbind(test1[, which(colnames(test1) == 'time')], test1[, which(colnames(test1) == 'status')], 
                                   oh_test_x_dis, sta_test_x_con))
  }
  
  colnames(train_pro) = colnames(test_pro) = c('time', 'status', paste('x', 1:ncol(train_pro[, -(1:2)]), sep = ''))
  out = list('train_pro' = train_pro, 'test_pro' = test_pro)
  return(out)
  
}

prepro_BJ_ELM = function(train, test, impute = 'no need', threshold = 5){
  
  #Function used to preprocess the training data and testing data for BJ-ELM, including the steps of imputing covariates with 
  #missing values by RSF, standardlizing continuous covariates and converting the discrete covariates into dummy variables.
  
  #ARGUMENTS
  #train: A dataframe with training data.
  #test: A dataframe with testing data.
  #impute: Optional, imputation method for imputing covariates with missing values. If 'no need', it will take imputation, 
  #       choosing 'RSF' will impute the covariates with missing values by RSF. Default to 'no need'. 
  #threshold: A integer specifying the threshold. Each column having less than 'threshold' different values 
  #           will be converted into a factor, otherwise, it will be converted into a numeric vector. 
  
  #RETURN
  #train_pro: A dataframe with preprocessed training dataset.
  #test_pro: A dataframe with preprocessed testing dataset.
  
  if(impute == 'no need'){
    train1 = train
    test1 = test
  }else if(impute == 'RSF'){
    RSF_impute = impute_RSF(train, test)
    imp_data = as.data.frame(rbind(RSF_impute$imp_train, RSF_impute$imp_test))
    imp_data1 = data_trans(imp_data, threshold)
    train1 = imp_data1[1:nrow(train), ]
    test1 = imp_data1[(1+nrow(train)):nrow(imp_data1), ]
  }
  
  train_x_con = select_if(train1[, which(colnames(train1) != 'status'&colnames(train1) != 'time')], is.numeric)
  test_x_con = select_if(test1[, which(colnames(test1) != 'status'&colnames(test1) != 'time')], is.numeric)
  z_score_standardlize = z_score(train_x_con = train_x_con, test_x_con = test_x_con)
  sta_train_x_con = z_score_standardlize$sta_train_x
  sta_test_x_con = z_score_standardlize$sta_test_x
  
  if(ncol(train_x_con) == ncol(train1[, -(1:2)])){
    train_pro = as.data.frame(cbind(train1[, which(colnames(train1) == 'time')], train1[, which(colnames(train1) == 'status')], 
                                    sta_train_x_con))
    test_pro = as.data.frame(cbind(test1[, which(colnames(test1) == 'time')], test1[, which(colnames(test1) == 'status')], 
                                   sta_test_x_con))
  }else{
    train_x_dis = select_if(train1[, which(colnames(train1) != 'status'&colnames(train1) != 'time')], is.factor)
    test_x_dis = select_if(test1[, which(colnames(test1) != 'status'&colnames(test1) != 'time')], is.factor)
    onehot = onehot_encoding(train_x_dis = train_x_dis, test_x_dis = test_x_dis)
    oh_train_x_dis = onehot$train_x_dis_onehot
    oh_test_x_dis = onehot$test_x_dis_onehot
    train_pro = as.data.frame(cbind(train1[, which(colnames(train1) == 'time')], train1[, which(colnames(train1) == 'status')], 
                                    oh_train_x_dis, sta_train_x_con))
    test_pro = as.data.frame(cbind(test1[, which(colnames(test1) == 'time')], test1[, which(colnames(test1) == 'status')], 
                                   oh_test_x_dis, sta_test_x_con))
  }
  colnames(train_pro) = colnames(test_pro) = c('time', 'status', paste('x', 1:ncol(train_pro[, -(1:2)]), sep = ''))
  out = list('train_pro' = train_pro, 'test_pro' = test_pro)
  return(out)
  
}


#2. Support functions for calculating IBS. -----------------------------------------------------------

cumsumsurv = function(x){

  #Function to compute jumps of Kapla-Meier given Lagrange multiplier lambda.
  
  if (any(is.na(x)))
    stop("NaNs")
  s = x
  .C("cumsumsurv", x = as.numeric(x), s = as.numeric(s), LLL = length(x))$s
}


locate = function(x, surv, time, logTt_fx_vec){
  
  #Function to calculate the predicted survival probability of a testing sample by K-M method based on
  #the survival function predicted by the model.
  
  if(is.na(x)){
    if(logTt_fx_vec %in% time){
      z = surv[which(time == logTt_fx_vec)[1]]
    }else{
      z = surv[which(time > logTt_fx_vec)[1]-1]}
  }else{
    z = x
  }
  return(z)
} 

pro_mat = function(T_train, f_x_train, status, f_x_test, dis_time){
  
  #Function to calculate a matrix of predicted values of survival probabilities for the testing set which will be used to calculate IBS.
  
  #ARGUMENTS
  #T_train: A vector of the logarithm of survival time in training dataset.
  #f_x_train: Estimated the logarithm of survival time from the training sample.
  #status: A vector of censoring indicator, must be 0 or 1 with 0=alive, 1=dead.
  #f_x_test: Estimated the logarithm of survival time from the testing sample.
  #dis_time: A vector of discrete time points.
  
  #RETURN
  #pro_mat: A matrix of predicted values of survival probabilities for the testing set. Rows denote different samples, columns denote  
  #         different time points, and the values in entry (i,j) of the matrix denote the predicted survival probability of the ith
  #         sample at the time point corresponding to the jth column.
  
  N = length(status)
  res = log(T_train)-f_x_train
  niceorder = order(res, - status)
  resorder = res[niceorder]
  dorder = status[niceorder]
  dorder[N] = 1
  
  temp = WKM(x = resorder, d = dorder, zc = 1:N)
  log_dis_time = log(dis_time)
  logTt_fx = matrix(rep(log_dis_time, length(f_x_test)), byrow = TRUE, ncol = length(log_dis_time))-
    matrix(rep(f_x_test, length(log_dis_time)), byrow = FALSE, ncol = length(log_dis_time))
  logTt_fx_vec = c(logTt_fx)
  survpro = rep(NA, length(logTt_fx_vec))
  survpro[which(logTt_fx_vec<min(temp$times))] = 1
  survpro[which(logTt_fx_vec>max(temp$times))] = 0
  survpro1 = imap_dbl(survpro, function(.x, .y) locate(x = .x, surv = temp$surv, time = temp$times, logTt_fx_vec = logTt_fx_vec[.y]))
  pro_mat = matrix(survpro1, ncol = ncol(logTt_fx), byrow = FALSE)
  return(pro_mat)
}


#3. K-folds cross validation for RSF -------------------------------------------------------
CV_RSF = function(data, k_folds, mtry, seed){
  
  #Function to take k-folds CV for RSF.
  
  #ARGUMENTS
  #data: A dataframe with training data.
  #k_folds: Number of fold of cv. Default to 5.
  #mtry: Optional user-supplied mtry parameter sequence.
  
  #RETURN
  #The best mtry parameter chosen by k-fold CV.
  
  data = data[, c(which(colnames(data) == 'time'), which(colnames(data) == 'status'), 
                  which(colnames(data) != 'status'&colnames(data) != 'time'))]
  tmp1 = expand.grid(mtry)
  tmp1 = cbind( tmp1[, 1])
  nn1 = dim(tmp1)[1]
  C_index = matrix(NA, nrow = nn1, ncol = k_folds)
  set.seed(seed = seed)
  folds = createFolds(y = data$time, k = k_folds)
  
  for(k in 1:k_folds){
    test = data[folds[[k]], ]
    train = data[-folds[[k]], ]
    
    
    for (i in 1:nn1) {
      formula2 = as.formula(paste('Surv(time, status)~', paste(colnames(train)[-(1:2)], collapse = "+")))
      RSFS = rfsrc(formula2, data = train, splitrule = "logrankscore", mtry = tmp1[i, 1], 
                   na.action = "na.impute")
      RSFS_predict = predict(object = RSFS, newdata = test)
      C_index[i, k] = estC(timeX = test$time, statusX = test$status, scoreY = exp(-RSFS_predict[["predicted"]]))
    }
  }
  C_index_out = apply(C_index, 1, mean)
  max_ind = which.max(C_index_out)
  best_mtry = tmp1[max_ind, 1]
  out = list('best_mtry' = best_mtry)
  return(out)
}



