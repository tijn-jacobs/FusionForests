################################################################################
#
#                                                                                     
#   Filename    :    Simulations.R    												  
#   Project     :    Article "Buckley-James Boosting Model based on Extreme
#                    Learning Machine and Random Survival Forests"                                                             
#   Authors     :    Jianfen Kong, Shuhong Zhang                                                              
#   Date        :    20/02/2023
#                    Take the experiments of simulations (Section 4.2 in the article.) 
#                      to produce table 1 and table 2 which shows the results of simulation studies.
#   Input data files  :    ---                                                        
#   Output data files :    BJ_ELM_Code/RESULT/Simulations(Table_1).xlsx,
#                          BJ_ELM_Code/RESULT/Simulations(Table_2).xlsx
#                          BJ_ELM_Code/RESULT/Simulations_Studies_Results.Rdata
#   R Version   :    R-4.1.3                                                              
#   Required R packages :  bujar_2.1.8, gbm.30, writexl_1.4.0 and VIM_6.1.1.
#
################################################################################
start_time = Sys.time()

source('BJ_ELM_Code/R/Main_Functions/ELM_Boosting.R')
source('BJ_ELM_Code/R/Main_Functions/BJ_ELM.R')
source('BJ_ELM_Code/R/Main_Functions/Auxiliary_Functions.R')
source('BJ_ELM_Code/R/Simulation_studies/Simulate_Survival_Datasets.R')

library('bujar')
library('gbm')
library('VIM')
library('writexl')
library('survival')
library('compositions')


# Define functions to provide framework for the experiments ---------------

#(1)Without Missing

SIM_WITHOUT_MISSING = function(n, nbase, nhid, nite, seed = 1, simulate, skip = NULL){
  
  #Funtion to provide a overall framework for the experiments whose simulated survival data without missing values on covariates. 
  
  #ARGUMENTS
  #n: Number of samples in each simulated survival dataset.
  #nbase: Optional user-supplied Mstop sequence. Default to (10, 15, 20, 25, 30, 35).
  #nhid: Optional user-supplied sequence of the number of hidden neurons. Default to (5, 10, 15, 20, 25, 30).
  #nite: Number of times the experiment was repeated.
  #seed: A numeric value specifying the random seed. Defaults to 1.
  #simulate: Optional, 'linear' for the experiments of simulating linear effects survival data, 
  #          'nonlinear' for the experiments of simulating non-linear effects survival data.  
  
  #RETURN
  #C_index: A dataframe with C-index values of six models in 20 times experiments.
  #best_nbase: A vector of the best Mstop chosen by k-fold CV in 20 times experiments.
  #best_nhid: A vector of the best nhid chosen by k-fold CV in 20 times experiments.
  #censoring_rate: A vector of censoring rate of 20 simulated survival datasets.
  #time_consumed: A dataframe with running time of six models in 20 times experiments.
  #IBS: A dataframe with IBS values of six models in 20 times experiments.
  
  
  C_BJ_LS = C_BJ_Trees = C_cph = 
    C_RSF = C_BJ = C_BJ_ELM = c()
  
  IBS_BJ_LS = IBS_BJ_Trees = IBS_cph = 
    IBS_RSF = IBS_BJ = IBS_BJ_ELM = c()
  
  time_BJ_LS = time_BJ_Trees = time_cph = 
    time_RSF = time_BJ = time_BJ_ELM = c()
  cen_rate = best_nbase = best_nhid = best_mtry = c()
  
  for(i in 1:nite){
    cat('iteration', i, "\n")

    #Step 1. Simulate linear effects survival data.
    if((i+seed) %in% skip){
      set.seed(i+seed+1)
    }else{
      set.seed(i+seed)
    }
    if(simulate == 'linear'){
      sim = pro_surv_linear(n = n)
      
    }else if(simulate == 'nonlinear'){
      sim = pro_surv_nonlinear(n = n)
      
    }
    data = sim$data
    colname_data = colnames(data)
    cen_rate[i] = sim$censoring_rate
    cat('censoring', i, cen_rate[i],  "\n")
    
    #Step 2. Split data into training data and testing data.
    set.seed(i + seed)
    test_ind = sample(1:n, round(0.2*n))
    train = data[-test_ind, ]
    test = data[test_ind, ]
    
    #Step 3. Preprocess training data and testing data.
    prepro_common = prepro_common(train, test)
    train_pro = prepro_common$train_pro
    test_pro = prepro_common$test_pro
    surv_obj_pro = Surv(test_pro$time, test_pro$status)
    
    formula1 = as.formula(paste("Surv(time,  status)~",  paste(colnames(train_pro[-(1:2)]),  collapse = "+")))
    
    #Step 4. Models training and predictions.
    ##(1)RSF
    RSF_start_time = Sys.time()
    cv_RSF = CV_RSF(data = train_pro, k_folds = 5, 
                    mtry = seq(floor(ncol(train_pro[, -(1:2)]))/2, ncol(train_pro[, -(1:2)]), 1), seed = seed+i)
    best_mtry[i] = cv_RSF$best_mtry
    RSF = rfsrc(formula1,  data = train_pro, splitrule = "logrankscore", 
                mtry = best_mtry[i], na.action = "na.impute")
    dis_time = RSF$time.interest
    RSF_predict = predict(object = RSF, newdata = test_pro)
    RSF_end_time = Sys.time()
    time_RSF[i] = difftime(RSF_end_time, RSF_start_time, units = "secs")
    C_RSF[i] = estC(timeX = test_pro$time,  statusX = test_pro$status,  scoreY = exp(-RSF_predict[["predicted"]]))
    cat('C_RSF', i, C_RSF[i],  "\n")
    RSF_mat = predict(RSF,  test_pro)$survival
    IBS_RSF[i] = IBS(surv_obj_pro,  sp_matrix = RSF_mat,  IBSrange = dis_time)
    cat('IBS_RSF', i, IBS_RSF[i],  "\n")
    
    ##(2)BJ_ELM
    BJ_ELM_start_time = Sys.time()
    cv_BJ_ELM = CV_BJ_ELM(data = train_pro, k_folds = 5, nbase = nbase, nhid = nhid, seed = seed+i)
    best_nbase[i] = cv_BJ_ELM$best_nbase
    best_nhid[i] = cv_BJ_ELM$best_nhid
    if((i+seed) %in% skip){
      set.seed(i+seed+1)
    }else{
      set.seed(i+seed)
    }
    BJ_ELM = BJ_ELM(x = train_pro[, -(1:2)],  y = log(train_pro$time),  status = train_pro$status, 
                    nhid = best_nhid[i], nbase = best_nbase[i])
    y_pre = pre_BJ_ELM(model = BJ_ELM$model, newx = as.matrix(test_pro[, -(1:2)]))
    BJ_ELM_end_time = Sys.time()
    time_BJ_ELM[i] = difftime(BJ_ELM_end_time, BJ_ELM_start_time, units = "secs")
    C_BJ_ELM[i] = estC(timeX = test_pro$time,  statusX = test_pro$status,  scoreY = exp(y_pre))
    cat('C_BJ_ELM', i, C_BJ_ELM[i],  "\n")
    BJ_ELM_mat = pro_mat(T_train = train_pro$time, f_x_train = BJ_ELM$model$y_hat, status = train_pro$status, 
                         f_x_test = y_pre, dis_time = dis_time)
    IBS_BJ_ELM[i] = IBS(surv_obj_pro,  sp_matrix = BJ_ELM_mat,  IBSrange = dis_time)
    cat('IBS_BJ_ELM', i, IBS_BJ_ELM[i],  "\n")
    
    ##(3)cph
    cph_start_time = Sys.time()
    cph = coxph(formula1,  data = train_pro, x = TRUE)
    cph_end_time = Sys.time()
    C_cph[i] = concordance(object = cph, newdata = test_pro)$concordance
    time_cph[i] = difftime(cph_end_time, cph_start_time, units = "secs")
    cat('C_cph', i, C_cph[i],  "\n")
    cox_mat = predictSurvProb(cph,  newdata = test_pro,  times = dis_time)
    IBS_cph[i] = IBS(surv_obj_pro,  sp_matrix = cox_mat,  IBSrange = dis_time)
    cat('IBS_cph', i, IBS_cph[i],  "\n")
    
    pred_cph=predict(cph, newdata = test_pro,
                     type="expected")
    C_cph1= estC(timeX = test_pro$time,  statusX = test_pro$status,  scoreY = exp(-pred_cph))
      
    ##(4)BJ
    BJ_start_time = Sys.time()
    BJ = bujar(y = log(train_pro$time),  cens = train_pro$status,  x = train_pro[, -(1:2)], 
               learner = "enet", tuning = TRUE, cv = TRUE, nfold = 5, 
               lamb = 0,  s = 1, center = FALSE) 
    pred_BJ = predict(BJ,  newx = test_pro[, -(1:2)])
    pred_BJ = exp(pred_BJ)
    BJ_end_time = Sys.time()
    C_BJ[i] = estC(timeX = test_pro$time,  statusX = test_pro$status,  scoreY = pred_BJ)
    time_BJ[i] = difftime(BJ_end_time, BJ_start_time, units = "secs")
    cat('C_BJ', i, C_BJ[i],  "\n")
    BJ_mat = pro_mat(T_train = train_pro$time, f_x_train = BJ$yhat, status = train_pro$status, 
                     f_x_test = log(pred_BJ), dis_time = dis_time)
    IBS_BJ[i] = IBS(surv_obj_pro,  sp_matrix = BJ_mat,  IBSrange = dis_time)
    cat('IBS_BJ', i, IBS_BJ[i],  "\n")
    
    ##(5)BJ_LS
    BJ_LS_start_time = Sys.time()
    BJ_LS = bujar(y = log(train_pro$time), cens = train_pro$status, x = train_pro[, -(1:2)], 
                  tuning = TRUE, cv = TRUE, nfold = 5, vimpint = FALSE, center = FALSE)  
    pred_BJ_LS = predict(BJ_LS,  newx = as.matrix(test_pro[, -(1:2)]))
    pred_BJ_LS = exp(pred_BJ_LS)
    BJ_LS_end_time = Sys.time()
    C_BJ_LS[i] = estC(timeX = test_pro$time,  statusX = test_pro$status,  scoreY = pred_BJ_LS)
    time_BJ_LS[i] = difftime(BJ_LS_end_time, BJ_LS_start_time, units = "secs")
    cat('C_BJ_LS', i, C_BJ_LS[i],  "\n")
    BJ_LS_mat = pro_mat(T_train = train_pro$time, f_x_train = BJ_LS$yhat, 
                        status = train_pro$status, 
                        f_x_test = log(pred_BJ_LS), dis_time = dis_time)
    IBS_BJ_LS[i] = IBS(surv_obj_pro,  sp_matrix = BJ_LS_mat,  IBSrange = dis_time)
    cat('IBS_BJ_LS', i, IBS_BJ_LS[i],  "\n")
    
    ##(6)BJ_Trees
    BJ_Trees_start_time = Sys.time()
    BJ_Trees = bujar(y = log(train_pro$time), cens = train_pro$status, x = train_pro[, -(1:2)], learner = "tree", 
                     tuning = TRUE, cv = TRUE, nfold = 5, vimpint = FALSE, center = FALSE)
    pred_BJ_Trees = predict(BJ_Trees,  newx = as.matrix(test_pro[, -(1:2)]))
    pred_BJ_Trees = exp(pred_BJ_Trees)
    BJ_Trees_end_time = Sys.time()
    C_BJ_Trees[i] = estC(timeX = test_pro$time,  statusX = test_pro$status,  scoreY = pred_BJ_Trees)
    time_BJ_Trees[i] = difftime(BJ_Trees_end_time, BJ_Trees_start_time, units = "secs")
    cat('C_BJ_Trees', i, C_BJ_Trees[i],  "\n")
    
    BJ_Trees_mat = pro_mat(T_train = train_pro$time, f_x_train = BJ_Trees$yhat, 
                           status = train_pro$status, 
                           f_x_test = log(pred_BJ_Trees), dis_time = dis_time)
    IBS_BJ_Trees[i] = IBS(surv_obj_pro,  sp_matrix = BJ_Trees_mat,  IBSrange = dis_time)
    cat('IBS_BJ_Trees', i, IBS_BJ_Trees[i],  "\n")
    
  }
  C_index = cbind(C_BJ_ELM, C_BJ, C_BJ_LS, C_BJ_Trees, C_cph, C_RSF)
  colnames(C_index) = c('C_BJ_ELM', 'C_BJ', 'C_BJ_LS', 'C_BJ_Trees', 'C_cph', 'C_RSF')
  
  IBS = cbind(IBS_BJ_ELM, IBS_BJ, IBS_BJ_LS, IBS_BJ_Trees, IBS_cph, IBS_RSF)
  colnames(IBS) = c('IBS_BJ_ELM', 'IBS_BJ', 'IBS_BJ_LS', 'IBS_BJ_Trees', 'IBS_cph', 'IBS_RSF')
  
  time_consumed = cbind(time_BJ_ELM, time_BJ, time_BJ_LS, time_BJ_Trees, time_cph, time_RSF)
  colnames(time_consumed) = c('time_BJ_ELM', 'time_BJ', 'time_BJ_LS', 'time_BJ_Trees', 'time_cph', 'time_RSF')
  
  out = list('C_index' = C_index, 'best_nbase' = best_nbase, 'best_nhid' = best_nhid, 'censoring_rate' = cen_rate, 
             'time_consumed' = time_consumed, 'IBS' = IBS)
  
  return(out)
}

#(2)With Missing

SIM_WITH_MISSING = function(n, NA_rate, nbase, nhid, nite, seed = 1, simulate, skip = NULL){
  
  #Funtion to provide a overall framework for the experiments whose simulated survival data with missing values on covariates. 
  
  #ARGUMENTS
  #n: Number of samples in each simulated survival dataset
  #NA_rate: A numeric value ranging from 0 to 1 to specify the missing rate of covariates.
  #nbase: Optional user-supplied Mstop sequence. Default to (10, 15, 20, 25, 30, 35).
  #nhid: Optional user-supplied sequence of the number of hidden neurons. Default to (5, 10, 15, 20, 25, 30).
  #nite: Number of times the experiment was repeated.
  #seed: A numeric value specifying the random seed. Defaults to 1
  #simulate: Optional, 'linear' for the experiments of simulating linear effects survival data with missing values, 
  #          'nonlinear' for the experiments of simulating non-linear effects survival data with missing values. 
  
  #RETURN
  #C_index: A dataframe with C-index values of six models in 20 times experiments.
  #best_nbase: A vector of the best Mstop chosen by k-fold CV in 20 times experiments.
  #best_nhid: A vector of the best nhid chosen by k-fold CV in 20 times experiments.
  #censoring_rate: A vector of censoring_rate of 20 simulated survival datasets.
  #time_consumed: A dataframe with running time of six models in 20 times experiments.
  #IBS: A dataframe with IBS values of six models in 20 times experiments  
  
  C_BJ_LS = C_BJ_Trees = C_cph = 
    C_RSF = C_BJ = C_BJ_ELM = c()
  
  IBS_BJ_LS = IBS_BJ_Trees = IBS_cph = 
    IBS_RSF = IBS_BJ = IBS_BJ_ELM = c()
  
  time_BJ_LS = time_BJ_Trees = time_cph = 
    time_RSF = time_BJ = time_BJ_ELM = c()
  
  cen_rate = best_nbase = best_nhid = best_mtry = c()
  
  for(i in 1:nite){
    cat('iteration', i, "\n")
    
    #Step 1. Simulate linear effects survival data with missing values on covariates.
    if((i+seed) %in% skip){
      set.seed(i+seed+1)
    }else{
      set.seed(i+seed)
    }
    if(simulate == 'linear'){
      sim = pro_surv_linear_miss(n = n, NA_rate = NA_rate)
      
    }else if(simulate == 'nonlinear'){
      sim = pro_surv_nonlinear_miss(n = n, NA_rate = NA_rate)
      
    }
    data = sim$data
    colname_data = colnames(data)
    cen_rate[i] = sim$censoring_rate
    cat('censoring', i, cen_rate[i],  "\n")
    
    #Step 2. Split data into training data and testing data.
    set.seed(i+seed)
    test_ind = sample(1:n, round(0.2*n))
    train = data[-test_ind, ]
    test = data[test_ind, ]
    
    #Step 3. Preprocess training data and testing data.
    prepro_common = prepro_common(train = train, test = test, impute = 'mean_mode', threshold = 2)
    train_pro = prepro_common$train_pro
    test_pro = prepro_common$test_pro
    prepro_BJ_ELM = prepro_BJ_ELM(train, test, impute = 'RSF', threshold = 2)
    train_BJ_ELM = prepro_BJ_ELM$train_pro
    test_BJ_ELM = prepro_BJ_ELM$test_pro
    
    surv_obj_pro = Surv(test_pro$time, test_pro$status)
    surv_obj_BJ_ELM = Surv(test_BJ_ELM$time, test_BJ_ELM$status)
    
    formula1 = as.formula(paste("Surv(time,  status)~",  paste(colnames(train_pro[-(1:2)]),  collapse = "+")))
    
    #Step 4. Models training and predictions.
    ##(1)RSF
    RSF_start_time = Sys.time()
    cv_RSF = CV_RSF(data = train_pro, k_folds = 5, 
                    mtry = seq(floor(ncol(train_pro[, -(1:2)]))/2, ncol(train_pro[, -(1:2)]), 1), seed = seed+i)
    best_mtry[i] = cv_RSF$best_mtry
    RSF = rfsrc(formula1,  data = train_pro, splitrule = "logrankscore", 
                mtry = best_mtry[i], na.action = "na.impute")
    dis_time = RSF$time.interest
    RSF_predict = predict(object = RSF, newdata = test_pro)
    RSF_end_time = Sys.time()
    time_RSF[i] = difftime(RSF_end_time, RSF_start_time, units = "secs")
    C_RSF[i] = estC(timeX = test_pro$time,  statusX = test_pro$status,  scoreY = exp(-RSF_predict[["predicted"]]))
    cat('C_RSF', i, C_RSF[i],  "\n")
    RSF_mat = predict(RSF,  test_pro)$survival
    IBS_RSF[i] = IBS(surv_obj_pro,  sp_matrix = RSF_mat,  IBSrange = dis_time)
    cat('IBS_RSF', i, IBS_RSF[i],  "\n")
    
    ##(2)BJ_ELM
    BJ_ELM_start_time = Sys.time()
    cv_BJ_ELM = CV_BJ_ELM(data = train_BJ_ELM, k_folds = 5, nbase = nbase, nhid = nhid, seed = seed+i)
    best_nbase[i] = cv_BJ_ELM$best_nbase
    best_nhid[i] = cv_BJ_ELM$best_nhid
    if((i+seed) %in% skip){
      set.seed(i+seed+1)
    }else{
      set.seed(i+seed)
    }    
    BJ_ELM = BJ_ELM(x = train_BJ_ELM[, -(1:2)],  y = log(train_BJ_ELM$time),  status = train_BJ_ELM$status, 
                    nhid = best_nhid[i], nbase = best_nbase[i])
    y_pre = pre_BJ_ELM(model = BJ_ELM$model, newx = as.matrix(test_BJ_ELM[, -(1:2)]))
    BJ_ELM_end_time = Sys.time()
    C_BJ_ELM[i] = estC(timeX = test_BJ_ELM$time,  statusX = test_BJ_ELM$status,  scoreY = exp(y_pre))
    time_BJ_ELM[i] = difftime(BJ_ELM_end_time, BJ_ELM_start_time, units = "secs")
    cat('C_BJ_ELM', i, C_BJ_ELM[i],  "\n")
    BJ_ELM_mat = pro_mat(T_train = train_BJ_ELM$time, f_x_train = BJ_ELM$model$y_hat, status = train_BJ_ELM$status, 
                         f_x_test = y_pre, dis_time = dis_time)
    IBS_BJ_ELM[i] = IBS(surv_obj_BJ_ELM,  sp_matrix = BJ_ELM_mat,  IBSrange = dis_time)
    cat('IBS_BJ_ELM', i, IBS_BJ_ELM[i],  "\n")
    
    ##(3)cph
    cph_start_time = Sys.time()
    cph = coxph(formula1,  data = train_pro, x = TRUE)
    cph_end_time = Sys.time()
    C_cph[i] = concordance(object = cph, newdata = test_pro)$concordance
    time_cph[i] = difftime(cph_end_time, cph_start_time, units = "secs")
    cat('C_cph', i, C_cph[i],  "\n")
    cox_mat = predictSurvProb(cph, newdata = test_pro, times = dis_time)
    IBS_cph[i] = IBS(surv_obj_pro, sp_matrix = cox_mat, IBSrange = dis_time)
    cat('IBS_cph', i, IBS_cph[i],  "\n")
    
    ##(4)BJ
    BJ_start_time = Sys.time()
    BJ = bujar(y = log(train_pro$time),  cens = train_pro$status,  x = train_pro[, -(1:2)], 
               learner = "enet", tuning = TRUE, cv = TRUE, nfold = 5, 
               lamb = 0,  s = 1, center = FALSE) 
    pred_BJ = predict(BJ,  newx = test_pro[, -(1:2)])
    pred_BJ = exp(pred_BJ)
    BJ_end_time = Sys.time()
    C_BJ[i] = estC(timeX = test_pro$time,  statusX = test_pro$status,  scoreY = pred_BJ)
    time_BJ[i] = difftime(BJ_end_time, BJ_start_time, units = "secs")
    cat('C_BJ', i, C_BJ[i],  "\n")
    BJ_mat = pro_mat(T_train = train_pro$time, f_x_train = BJ$yhat, status = train_pro$status, 
                     f_x_test = log(pred_BJ), dis_time = dis_time)
    IBS_BJ[i] = IBS(surv_obj_pro,  sp_matrix = BJ_mat,  IBSrange = dis_time)
    cat('IBS_BJ', i, IBS_BJ[i],  "\n")
    
    ##(5)BJ_LS
    BJ_LS_start_time = Sys.time()
    BJ_LS = bujar(y = log(train_pro$time), cens = train_pro$status, x = train_pro[, -(1:2)], 
                  tuning = TRUE, cv = TRUE, nfold = 5, vimpint = FALSE, center = FALSE)  
    pred_BJ_LS = predict(BJ_LS,  newx = as.matrix(test_pro[, -(1:2)]))
    pred_BJ_LS = exp(pred_BJ_LS)
    BJ_LS_end_time = Sys.time()
    C_BJ_LS[i] = estC(timeX = test_pro$time,  statusX = test_pro$status,  scoreY = pred_BJ_LS)
    time_BJ_LS[i] = difftime(BJ_LS_end_time, BJ_LS_start_time, units = "secs")
    cat('C_BJ_LS', i, C_BJ_LS[i],  "\n")
    BJ_LS_mat = pro_mat(T_train = train_pro$time, f_x_train = BJ_LS$yhat, 
                        status = train_pro$status, 
                        f_x_test = log(pred_BJ_LS), dis_time = dis_time)
    IBS_BJ_LS[i] = IBS(surv_obj_pro,  sp_matrix = BJ_LS_mat,  IBSrange = dis_time)
    cat('IBS_BJ_LS', i, IBS_BJ_LS[i],  "\n")
    
    ##(6)BJ_Trees
    BJ_Trees_start_time = Sys.time()
    BJ_Trees = bujar(y = log(train_pro$time), cens = train_pro$status, x = train_pro[, -(1:2)], learner = "tree", 
                     tuning = TRUE, cv = TRUE, nfold = 5, vimpint = FALSE, center = FALSE)
    pred_BJ_Trees = predict(BJ_Trees,  newx = as.matrix(test_pro[, -(1:2)]))
    pred_BJ_Trees = exp(pred_BJ_Trees)
    BJ_Trees_end_time = Sys.time()
    C_BJ_Trees[i] = estC(timeX = test_pro$time,  statusX = test_pro$status,  scoreY = pred_BJ_Trees)
    time_BJ_Trees[i] = difftime(BJ_Trees_end_time, BJ_Trees_start_time, units = "secs")
    cat('C_BJ_Trees', i, C_BJ_Trees[i],  "\n")
    BJ_Trees_mat = pro_mat(T_train = train_pro$time, f_x_train = BJ_Trees$yhat, 
                           status = train_pro$status, 
                           f_x_test = log(pred_BJ_Trees), dis_time = dis_time)
    IBS_BJ_Trees[i] = IBS(surv_obj_pro,  sp_matrix = BJ_Trees_mat,  IBSrange = dis_time)
    cat('IBS_BJ_Trees', i, IBS_BJ_Trees[i],  "\n")
    
  }
  
  C_index = cbind(C_BJ_ELM, C_BJ, C_BJ_LS, C_BJ_Trees, C_cph, C_RSF)
  colnames(C_index) = c('C_BJ_ELM', 'C_BJ', 'C_BJ_LS', 'C_BJ_Trees', 'C_cph', 'C_RSF')
  
  IBS = cbind(IBS_BJ_ELM, IBS_BJ, IBS_BJ_LS, IBS_BJ_Trees, IBS_cph, IBS_RSF)
  colnames(IBS) = c('IBS_BJ_ELM', 'IBS_BJ', 'IBS_BJ_LS', 'IBS_BJ_Trees', 'IBS_cph', 'IBS_RSF')
  
  time_consumed = cbind(time_BJ_ELM, time_BJ, time_BJ_LS, time_BJ_Trees, time_cph, time_RSF)
  colnames(time_consumed) = c('time_BJ_ELM', 'time_BJ', 'time_BJ_LS', 'time_BJ_Trees', 'time_cph', 'time_RSF')
  
  out = list('C_index' = C_index, 'best_nbase' = best_nbase, 'best_nhid' = best_nhid, 'censoring_rate' = cen_rate, 
             'time_consumed' = time_consumed, 'IBS' = IBS)
  
  return(out)
}

#1. Linear ------------------------------------------------------------------

#(1)Take the experiment of simulated linear survival data without missing values.
sim_linear = SIM_WITHOUT_MISSING(n = 250, nbase = seq(10, 31, by = 3), nhid = seq(5, 11, by = 3), 
                                 nite = 20, seed = 16, simulate = 'linear')
#n = 250: Generate 250 samples in each simulated linear effects survival dataset (n = 250).
#nbase = seq(10, 31, by = 3): The best Mstop (nbase) is chosen from sequence (10, 13, 16, 19, 22, 25, 28, 31) by 5-fold CV.
#nhid = seq(5, 11, by = 3): Tthe best number of hidden nodes (nhid) is chosen from sequence (5, 8, 11, 11) by 5-fold CV.

#(2)Obtain results
#Object 'linear_C_index' and 'linear_IBS' have saved the C-index and IBS of six models on simulated linear survival data
#without missing values showed in table 1.

#Object 'linear_time' have saved the running time of BJ-ELM, BJ-LS and BJ-Trees on simulated linear survival data 
#without missing values showed in table 2. 
linear_C_index = data.frame(
  'Model' = c('BJ-ELM', 'BJ', 'BJ-LS', 'BJ-Trees', 'Coxph', 'RSFS'), 
  'means' = round(c(apply(sim_linear$C_index, 2, mean)), 3), 
  "CI lower" = round(c(apply(sim_linear$C_index, 2, mean)-1.96*apply(sim_linear$C_index, 2, sd)/sqrt(20)), 3), 
  "CI upper" = round(c(apply(sim_linear$C_index, 2, mean)+1.96*apply(sim_linear$C_index, 2, sd)/sqrt(20)), 3)
  
)

linear_IBS = data.frame(
  'Model' = c('BJ-ELM', 'BJ', 'BJ-LS', 'BJ-Trees', 'Coxph', 'RSFS'), 
  'means' = round(c(apply(sim_linear$IBS, 2, mean)), 3), 
  "CI lower" = round(c(apply(sim_linear$IBS, 2, mean)-1.96*apply(sim_linear$IBS, 2, sd)/sqrt(20)), 3), 
  "CI upper" = round(c(apply(sim_linear$IBS, 2, mean)+1.96*apply(sim_linear$IBS, 2, sd)/sqrt(20)), 3)
  
)


#2. Non-linear --------------------------------------------------------------

sim_nonlinear = SIM_WITHOUT_MISSING(n = 250, nbase = seq(16, 25, by = 3), nhid = seq(19, 25, by = 3), 
                                    nite = 20, seed = 25, simulate = 'nonlinear', skip=c(32, 37))

nonlinear_C_index = data.frame(
  'Model' = c('BJ-ELM', 'BJ', 'BJ-LS', 'BJ-Trees', 'Coxph', 'RSFS'), 
  'means' = round(c(apply(sim_nonlinear$C_index, 2, mean)), 3), 
  "CI lower" = round(c(apply(sim_nonlinear$C_index, 2, mean)-1.96*apply(sim_nonlinear$C_index, 2, sd)/sqrt(20)), 3), 
  "CI upper" = round(c(apply(sim_nonlinear$C_index, 2, mean)+1.96*apply(sim_nonlinear$C_index, 2, sd)/sqrt(20)), 3)
  
)

nonlinear_IBS = data.frame(
  'Model' = c('BJ-ELM', 'BJ', 'BJ-LS', 'BJ-Trees', 'Coxph', 'RSFS'), 
  'means' = round(c(apply(sim_nonlinear$IBS, 2, mean)), 3), 
  "CI lower" = round(c(apply(sim_nonlinear$IBS, 2, mean)-1.96*apply(sim_nonlinear$IBS, 2, sd)/sqrt(20)), 3), 
  "CI upper" = round(c(apply(sim_nonlinear$IBS, 2, mean)+1.96*apply(sim_nonlinear$IBS, 2, sd)/sqrt(20)), 3)
  
)

#3. Linear with missing --------------------------------

##(1)Take the experiments.
sim_linear_missing = SIM_WITH_MISSING(n = 250, NA_rate = 0.2, nbase = seq(20, 32, by = 3), nhid = seq(5, 11, 3), 
                                      nite = 20, seed = 3, simulate = 'linear') 

#NA_rate = 0.2: In each simulated dataset,  we delete 20% data of covariates based on the assumption of missing at random (MAR).

#(2)Obtain results.
linear_missing_C_index = data.frame(
  'Model' = c('BJ-ELM', 'BJ', 'BJ-LS', 'BJ-Trees', 'Coxph', 'RSFS'), 
  'means' = round(c(apply(sim_linear_missing$C_index, 2, mean)), 3), 
  "CI lower" = round(c(apply(sim_linear_missing$C_index, 2, mean)-1.96*apply(sim_linear_missing$C_index, 2, sd)/sqrt(20)), 3), 
  "CI upper" = round(c(apply(sim_linear_missing$C_index, 2, mean)+1.96*apply(sim_linear_missing$C_index, 2, sd)/sqrt(20)), 3)
  
)

linear_missing_IBS = data.frame(
  'Model' = c('BJ-ELM', 'BJ', 'BJ-LS', 'BJ-Trees', 'Coxph', 'RSFS'), 
  'means' = round(c(apply(sim_linear_missing$IBS, 2, mean)), 3), 
  "CI lower" = round(c(apply(sim_linear_missing$IBS, 2, mean)-1.96*apply(sim_linear_missing$IBS, 2, sd)/sqrt(20)), 3), 
  "CI upper" = round(c(apply(sim_linear_missing$IBS, 2, mean)+1.96*apply(sim_linear_missing$IBS, 2, sd)/sqrt(20)), 3)
  
)


#4. Non-linear with missing -----------------------------

##(1)Take the experiment
sim_nonlinear_missing = SIM_WITH_MISSING(n = 250, NA_rate = 0.2, nbase = seq(14, 28, 2), nhid = seq(5, 11, 3),
                                          nite = 20, seed = 3, simulate = 'nonlinear')  


#(2)Obtain results
nonlinear_missing_C_index = data.frame(
  'Model' = c('BJ-ELM', 'BJ', 'BJ-LS', 'BJ-Trees', 'Coxph', 'RSFS'), 
  'means' = round(c(apply(sim_nonlinear_missing$C_index, 2, mean)), 3), 
  "CI lower" = round(c(apply(sim_nonlinear_missing$C_index, 2, mean)-1.96*apply(sim_nonlinear_missing$C_index, 2, sd)/sqrt(20)), 3), 
  "CI upper" = round(c(apply(sim_nonlinear_missing$C_index, 2, mean)+1.96*apply(sim_nonlinear_missing$C_index, 2, sd)/sqrt(20)), 3)
  
)

nonlinear_missing_IBS = data.frame(
  'Model' = c('BJ-ELM', 'BJ', 'BJ-LS', 'BJ-Trees', 'Coxph', 'RSFS'), 
  'means' = round(c(apply(sim_nonlinear_missing$IBS, 2, mean)), 3), 
  "CI lower" = round(c(apply(sim_nonlinear_missing$IBS, 2, mean)-1.96*apply(sim_nonlinear_missing$IBS, 2, sd)/sqrt(20)), 3), 
  "CI upper" = round(c(apply(sim_nonlinear_missing$IBS, 2, mean)+1.96*apply(sim_nonlinear_missing$IBS, 2, sd)/sqrt(20)), 3)
  
)


# Output the results of the experiments of simulation studies. ------------

table1 = list("linear-C_index" = linear_C_index, 
              "linear-IBS" = linear_IBS, 
              "nonlinear-C_index" = nonlinear_C_index, 
              "nonlinear-IBS" = nonlinear_IBS, 
              "linear_missing-C_index" = linear_missing_C_index, 
              "linear_missing-IBS" = linear_missing_IBS, 
              "nonlinear_missing-C_index" = nonlinear_missing_C_index, 
              "nonlinear_missing-IBS" = nonlinear_missing_IBS
             )

table2 = data.frame("Dataset" = c('BJ-ELM', 'BJ-LS', 'BJ-Trees'), 
              "Linear" = round(c(apply(sim_linear$time_consumed[, c(1, 3, 4)], 2, mean)), 3), 
              "Non-linear" = round(c(apply(sim_nonlinear$time_consumed[, c(1, 3, 4)], 2, mean)), 3), 
              "Linear with missing" = round(c(apply(sim_linear_missing$time_consumed[, c(1, 3, 4)], 2, mean)), 3), 
              "Non-linear with missing" = round(c(apply(sim_nonlinear_missing$time_consumed[, c(1, 3, 4)], 2, mean)), 3)
)
End_time = Sys.time()

write_xlsx(table1, "BJ_ELM_Code/RESULT/Simulations_studies(Table_1).xlsx")
write_xlsx(table2, "BJ_ELM_Code/RESULT/Simulations_studies(Table_2).xlsx")
save.image('BJ_ELM_Code/RESULT/Simulations_Studies_Results.Rdata')

