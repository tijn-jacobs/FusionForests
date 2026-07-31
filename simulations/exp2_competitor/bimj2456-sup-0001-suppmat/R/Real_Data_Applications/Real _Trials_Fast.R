################################################################################
#
#                                                                                     
#   Filename    :    real trial.R    												  
#   Project     :    Article "Buckley-James Boosting Model based on 
#                    Extreme Learning Machine and Random Survival Forests" 
#   Authors     :    Jianfen Kong, Shuhong Zhang                                                              
#   Date        :    20/02/2022
#   Purpose     :    Take the experiments of real data applications (Section 4.3 in the article) 
#                    to produce table 3 and table 4 which show main information of six real datasets
#                    and the performance of six models on these six datasets respectively.
#   Input data files  :    BJ_ELM_Code/R/Real_Data_Applications/Best_Para_For_Real_Trials.RData                                                       
#   Output data files :    BJ_ELM_Code/RESULT/Real_Data_Applications(Table_3).xlsx,
#                          BJ_ELM_Code/RESULT/Real_Data_Applications(Table_4).xlsx
#                          BJ_ELM_Code/RESULT/Real_Data_Applications_Results.Rdata
#   R Version   :    R-4.1.3                                                              
#   Required R packages :  bujar_0.2-9, gbm.2.1.8 and writexl_1.4.0.
#
################################################################################


source('BJ_ELM_Code/R/Main_Functions/ELM_Boosting.R')
source('BJ_ELM_Code/R/Main_Functions/BJ_ELM.R')
source('BJ_ELM_Code/R/Main_Functions/Auxiliary_Functions.R')
load("BJ_ELM_Code/R/Real_Data_Applications/Best_Para_For_Real_Trials.RData")

library('bujar')
library('gbm')
library('writexl')


#Define a function to provide a overall framework for the experiments of real survival data
REAL_TRIAL=function(data, best_nhid, best_nbase, nite, 
                    impute_BJ_ELM = 'no need', impute_common = 'no need', 
                    threshold = 5, seed = 2, skip = NULL){
  
  #The function is developed to provide a overall framework for the experiments of real data applications.
  
  #ARGUMENTS
  #data: A dataframe with real survival data. The columns of survival time and censoring indicator
  #      should be named as 'time' and 'status' respectively.
  #best_nbase: A vector of the best Mstop chosen by k-fold CV in 20 times experiments.
  #best_nhid: A vector of the best nhid chosen by k-fold CV in 20 times experiments.
  #nite: Number of times the experiment was repeated.
  #seed: A numeric value specifying the random seed. Defaults to 1.
  #skip: A vector contains the seeds that not all five competing models could train model successfully 
  #      under the data splitting based on the seed.
  
  
  #RETURN
  #C_index: A dataframe with C-index values of six models in 20 times experiments.
  #IBS: A dataframe with IBS values of six models in 20 times experiments.
  
  data1 = data_trans(data, threshold = threshold)
  
  C_BJ_LS = C_BJ_Trees = C_cph = 
    C_RSF = C_BJ = C_BJ_ELM = c()
  
  IBS_BJ_LS = IBS_BJ_Trees = IBS_cph = 
    IBS_RSF = IBS_BJ = IBS_BJ_ELM = c()
  
  best_mtry = c()
  
  for(i in 1:nite){

    cat('iteration', i, "\n")
    
    #Step 1. Split data into training data and testing data.
    if((i+seed) %in% skip){
      set.seed(i+seed+1)
    }else{
      set.seed(i+seed)
    }
    test_ind = sample(1:nrow(data1), round(0.2*nrow(data1)))
    train = data1[-test_ind, ]
    test = data1[test_ind, ]
    
    #Step 2. Preprocess training data and testing data.
    prepro_common = prepro_common(train = train, test = test, impute = impute_common, threshold = threshold)
    train_pro = prepro_common$train_pro
    test_pro = prepro_common$test_pro
    prepro_BJ_ELM = prepro_BJ_ELM(train, test, impute = impute_BJ_ELM, threshold = threshold)
    train_BJ_ELM = prepro_BJ_ELM$train_pro
    test_BJ_ELM = prepro_BJ_ELM$test_pro
    surv_obj_pro = Surv(test_pro$time, test_pro$status)
    surv_obj_BJ_ELM = Surv(test_BJ_ELM$time, test_BJ_ELM$status)
    
    formula1 = as.formula(paste("Surv(time,  status)~",  paste(colnames(train_pro[-(1:2)]),  collapse = "+")))
    
    ##3.model
    ##(1)RSF
    cv_RSF = CV_RSF(data = train_pro, k_folds = 5, 
                    mtry = seq(floor(ncol(train_pro[, -(1:2)]))/2, ncol(train_pro[, -(1:2)]), 1), seed = seed+i)
    best_mtry[i] = cv_RSF$best_mtry
    RSF = rfsrc(formula1,  data = train_pro, splitrule = "logrankscore", 
                mtry = best_mtry[i], na.action  = "na.impute")
    dis_time = RSF$time.interest[which(RSF$time.interest>0)]
    RSF_predict = predict(object = RSF, newdata = test_pro)
    C_RSF[i] = estC(timeX = test_pro$time,  statusX = test_pro$status,  scoreY = exp(-RSF_predict[["predicted"]]))
    cat('C_RSF', i, C_RSF[i],  "\n")
    RSF_mat  =  predict(RSF,  test_pro)$survival
    IBS_RSF[i] = IBS(surv_obj_pro,  sp_matrix  =  RSF_mat,  IBSrange = dis_time)
    cat('IBS_RSF', i, IBS_RSF[i],  "\n")
    
    ##(2)BJ_ELM
    BJ_ELM = BJ_ELM(x = train_BJ_ELM[, -(1:2)],  y = log(train_BJ_ELM$time),  status = train_BJ_ELM$status, 
                    nhid = best_nhid[i], nbase = best_nbase[i])
    y_pre = pre_BJ_ELM(model = BJ_ELM$model, newx  =  as.matrix(test_BJ_ELM[, -(1:2)]))
    C_BJ_ELM[i] = estC(timeX = test_BJ_ELM$time,  statusX = test_BJ_ELM$status,  scoreY = exp(y_pre))
    cat('C_BJ_ELM', i, C_BJ_ELM[i],  "\n")
    BJ_ELM_mat = pro_mat(T_train = train_BJ_ELM$time, f_x_train = BJ_ELM$model$y_hat, status = train_BJ_ELM$status, 
                         f_x_test = y_pre, dis_time = dis_time)
    IBS_BJ_ELM[i] = IBS(surv_obj_BJ_ELM,  sp_matrix  =  BJ_ELM_mat,  IBSrange = dis_time)
    cat('IBS_BJ_ELM', i, IBS_BJ_ELM[i],  "\n")
    
    
    ##(3)cph
    cph = coxph(formula1,  data = train_pro, x = TRUE)
    C_cph[i] = concordance(object = cph, newdata = test_pro)$concordance
    cat('C_cph', i, C_cph[i],  "\n")
    
    cox_mat  =  predictSurvProb(cph, newdata = test_pro,  times = dis_time)
    IBS_cph[i] = IBS(surv_obj_pro, sp_matrix  = cox_mat,  IBSrange = dis_time)
    cat('IBS_cph', i, IBS_cph[i], "\n")
    
    ##(4)BJ
    BJ = bujar(y = log(train_pro$time),  cens = train_pro$status,  x = train_pro[, -(1:2)], 
               learner = "enet", tuning = TRUE, cv = TRUE, nfold = 5, 
               lamb =0, s = 1, center = FALSE) 
    pred_BJ <- predict(BJ, newx = test_pro[, -(1:2)])
    pred_BJ = exp(pred_BJ)
    C_BJ[i] = estC(timeX = test_pro$time, statusX = test_pro$status, scoreY = pred_BJ)
    cat('C_BJ', i, C_BJ[i], "\n")
    BJ_mat = pro_mat(T_train = train_pro$time, f_x_train = BJ$yhat, status = train_pro$status, 
                     f_x_test = log(pred_BJ), dis_time = dis_time)
    IBS_BJ[i] = IBS(surv_obj_pro, sp_matrix = BJ_mat, IBSrange = dis_time)
    cat('IBS_BJ', i, IBS_BJ[i], "\n")
    
    ##(5)BJ_LS
    BJ_LS = bujar(y = log(train_pro$time), cens = train_pro$status, x = train_pro[, -(1:2)], 
                  tuning = TRUE, cv = TRUE, nfold = 5, vimpint = FALSE, center = FALSE)  
    pred_BJ_LS <- predict(BJ_LS, newx = as.matrix(test_pro[, -(1:2)]))
    pred_BJ_LS = exp(pred_BJ_LS)
    C_BJ_LS[i] = estC(timeX = test_pro$time, statusX = test_pro$status, scoreY = pred_BJ_LS)
    cat('C_BJ_LS', i, C_BJ_LS[i], "\n")
    BJ_LS_mat = pro_mat(T_train = train_pro$time, f_x_train = BJ_LS$yhat, 
                        status = train_pro$status, 
                        f_x_test = log(pred_BJ_LS), dis_time = dis_time)
    IBS_BJ_LS[i] = IBS(surv_obj_pro, sp_matrix = BJ_LS_mat, IBSrange = dis_time)
    cat('IBS_BJ_LS', i, IBS_BJ_LS[i], "\n")
    
    ##(6)BJ_Trees
    BJ_Trees = bujar(y = log(train_pro$time), cens = train_pro$status, x = train_pro[, -(1:2)], learner = "tree", 
                     tuning = TRUE, cv = TRUE, nfold = 5, vimpint = FALSE, center = FALSE)
    pred_BJ_Trees <- predict(BJ_Trees, newx = as.matrix(test_pro[, -(1:2)]))
    pred_BJ_Trees = exp(pred_BJ_Trees)
    C_BJ_Trees[i] = estC(timeX = test_pro$time, statusX = test_pro$status, scoreY = pred_BJ_Trees)
    cat('C_BJ_Trees', i, C_BJ_Trees[i], "\n")
    BJ_Trees_mat = pro_mat(T_train = train_pro$time, f_x_train = BJ_Trees$yhat, 
                           status = train_pro$status, 
                           f_x_test = log(pred_BJ_Trees), dis_time = dis_time)
    
    IBS_BJ_Trees[i]=IBS(surv_obj_pro, sp_matrix = BJ_Trees_mat, IBSrange = dis_time)
    cat('IBS_BJ_Trees', i, IBS_BJ_Trees[i], "\n")
  }
  
  C_index = cbind(C_BJ_ELM, C_BJ, C_BJ_LS, C_BJ_Trees, C_cph, C_RSF)
  colnames(C_index) = c('C_BJ_ELM', 'C_BJ', 'C_BJ_LS', 'C_BJ_Trees', 'C_cph', 'C_RSF')
  
  IBS = cbind(IBS_BJ_ELM, IBS_BJ, IBS_BJ_LS, IBS_BJ_Trees, IBS_cph, IBS_RSF)
  colnames(IBS) = c('IBS_BJ_ELM', 'IBS_BJ', 'IBS_BJ_LS', 'IBS_BJ_Trees', 
                    'IBS_cph', 'IBS_RSF')
  
  out = list('C_index' = C_index, 'IBS' = IBS)
  
  return(out)
}



# Experiment for Lung --------------------------------------------------------------------

trial_lung = REAL_TRIAL(data = lung1, best_nhid = best_para_lung$best_nhid, best_nbase  = best_para_lung$best_nbase,
                         nite = 20, impute_BJ_ELM = 'RSF', impute_common = 'mean_mode', 
                        threshold = 3, seed = 1, skip = c(5, 18))


#Object 'lung_C_index' has saved the C-index of six models on lung data set showed in table 4. 
lung_C_index = data.frame(
  'Model' = c('BJ-ELM', 'BJ', 'BJ-LS', 'BJ-Trees', 'Coxph', 'RSF'), 
  'means' = round(c(apply(trial_lung$C_index, 2, mean)), 3), 
  "CI lower" = round(c(apply(trial_lung$C_index, 2, mean) - 1.96*apply(trial_lung$C_index, 2, sd)/sqrt(20)), 3), 
  "CI upper" = round(c(apply(trial_lung$C_index, 2, mean) + 1.96*apply(trial_lung$C_index, 2, sd)/sqrt(20)), 3)
  
)

#Object 'lung_IBS' has saved the IBS of six models on lung data set showed in table 4. 
lung_IBS = data.frame(
  'Model' = c('BJ-ELM', 'BJ', 'BJ-LS', 'BJ-Trees', 'Coxph', 'RSF'), 
  'means' = round(c(apply(trial_lung$IBS, 2, mean)), 3), 
  "CI lower" = round(c(apply(trial_lung$IBS, 2, mean) - 1.96*apply(trial_lung$IBS, 2, sd)/sqrt(20)), 3), 
  "CI upper" = round(c(apply(trial_lung$IBS, 2, mean) + 1.96*apply(trial_lung$IBS, 2, sd)/sqrt(20)), 3)
  
)


#Experiment for  Pbc ---------------------------------------------------------------------
data(pbc, package = "randomForestSRC")
colnames(pbc)[which(colnames(pbc) == 'days')] = 'time'
cen_pbc = 1 - sum(pbc$status)/nrow(pbc)

trial_pbc = REAL_TRIAL(data = pbc, best_nhid = best_para_pbc$best_nhid, best_nbase  = best_para_pbc$best_nbase,
                       nite = 20, impute_BJ_ELM = 'RSF', impute_common = 'mean_mode', 
                       threshold  =  3, seed  = 13)

### C-index
pbc_C_index =  data.frame(
  'Model' = c('BJ-ELM', 'BJ', 'BJ-LS', 'BJ-Trees', 'Coxph', 'RSF'), 
  'means'  =  round(c(apply(trial_pbc$C_index, 2, mean)), 3), 
  "CI lower"  = round(c(apply(trial_pbc$C_index, 2, mean) - 1.96*apply(trial_pbc$C_index, 2, sd)/sqrt(20)), 3), 
  "CI upper"  = round(c(apply(trial_pbc$C_index, 2, mean) + 1.96*apply(trial_pbc$C_index, 2, sd)/sqrt(20)), 3)
  
)

###IBS
pbc_IBS =  data.frame(
  'Model' = c('BJ-ELM', 'BJ', 'BJ-LS', 'BJ-Trees', 'Coxph', 'RSF'), 
  'means'  =  round(c(apply(trial_pbc$IBS, 2, mean)), 3), 
  "CI lower"  = round(c(apply(trial_pbc$IBS, 2, mean) - 1.96*apply(trial_pbc$IBS, 2, sd)/sqrt(20)), 3), 
  "CI upper"  = round(c(apply(trial_pbc$IBS, 2, mean) + 1.96*apply(trial_pbc$IBS, 2, sd)/sqrt(20)), 3)
  
)


#Experiment for WPBC --------------------------------------------------------------------

trial_wpbc = REAL_TRIAL(data = wpbc, best_nhid = best_para_wpbc$best_nhid, best_nbase = best_para_wpbc$best_nbase,
                        nite = 20, impute_BJ_ELM = 'RSF', impute_common = 'mean_mode', threshold  =  2, seed  = 29)
                        


###C-index
wpbc_C_index =  data.frame(
  'Model' = c('BJ-ELM', 'BJ', 'BJ-LS', 'BJ-Trees', 'Coxph', 'RSF'), 
  'means' = round(c(apply(trial_wpbc$C_index, 2, mean)), 3), 
  "CI lower" = round(c(apply(trial_wpbc$C_index, 2, mean) - 1.96*apply(trial_wpbc$C_index, 2, sd)/sqrt(20)), 3), 
  "CI upper" = round(c(apply(trial_wpbc$C_index, 2, mean) + 1.96*apply(trial_wpbc$C_index, 2, sd)/sqrt(20)), 3)
  
)

##IBS
wpbc_IBS =  data.frame(
  'Model' = c('BJ-ELM', 'BJ', 'BJ-LS', 'BJ-Trees', 'Coxph', 'RSF'), 
  'means' = round(c(apply(trial_wpbc$IBS, 2, mean)), 3), 
  "CI lower" = round(c(apply(trial_wpbc$IBS, 2, mean) - 1.96*apply(trial_wpbc$IBS, 2, sd)/sqrt(20)), 3), 
  "CI upper" = round(c(apply(trial_wpbc$IBS, 2, mean) + 1.96*apply(trial_wpbc$IBS, 2, sd)/sqrt(20)), 3)
  
)


#Experiment for stageC ------------------------------------------------------------------

trial_stagec = REAL_TRIAL(data = stagec, best_nhid = best_para_stagec$best_nhid, best_nbase = best_para_stagec$best_nbase,
                          nite = 20, impute_BJ_ELM = 'RSF', impute_common = 'mean_mode', 
                          threshold = 3, seed = 20)

###C-index
stagec_C_index = data.frame(
  'Model' = c('BJ-ELM', 'BJ', 'BJ-LS', 'BJ-Trees', 'Coxph', 'RSF'), 
  'means' = round(c(apply(trial_stagec$C_index, 2, mean)), 3), 
  "CI lower" = round(c(apply(trial_stagec$C_index, 2, mean) - 1.96*apply(trial_stagec$C_index, 2, sd)/sqrt(20)), 3), 
  "CI upper" = round(c(apply(trial_stagec$C_index, 2, mean) + 1.96*apply(trial_stagec$C_index, 2, sd)/sqrt(20)), 3)
  
)

##IBS
stagec_IBS = data.frame(
  'Model' = c('BJ-ELM', 'BJ', 'BJ-LS', 'BJ-Trees', 'Coxph', 'RSF'), 
  'means' = round(c(apply(trial_stagec$IBS, 2, mean)), 3), 
  "CI lower" = round(c(apply(trial_stagec$IBS, 2, mean) - 1.96*apply(trial_stagec$IBS, 2, sd)/sqrt(20)), 3), 
  "CI upper" = round(c(apply(trial_stagec$IBS, 2, mean) + 1.96*apply(trial_stagec$IBS, 2, sd)/sqrt(20)), 3)
  
)


#Experiment for veteran -----------------------------------------------------------------

data(cancer, package = 'survival')
veteran$prior[which(veteran$prior == 10)] = 1


trial_veteran = REAL_TRIAL(data = veteran, best_nhid = best_para_veteran$best_nhid, best_nbase = best_para_veteran$best_nbase,
                           nite = 20, threshold = 4, seed  = 51)



###C-index
veteran_C_index =  data.frame(
  'Model' = c('BJ-ELM', 'BJ', 'BJ-LS', 'BJ-Trees', 'Coxph', 'RSF'), 
  'means' = round(c(apply(trial_veteran$C_index, 2, mean)), 3), 
  "CI lower" = round(c(apply(trial_veteran$C_index, 2, mean) - 1.96*apply(trial_veteran$C_index, 2, sd)/sqrt(20)), 3), 
  "CI upper" = round(c(apply(trial_veteran$C_index, 2, mean) + 1.96*apply(trial_veteran$C_index, 2, sd)/sqrt(20)), 3)
  
)

##IBS
veteran_IBS =  data.frame(
  'Model' = c('BJ-ELM', 'BJ', 'BJ-LS', 'BJ-Trees', 'Coxph', 'RSF'), 
  'means' = round(c(apply(trial_veteran$IBS, 2, mean)), 3), 
  "CI lower" = round(c(apply(trial_veteran$IBS, 2, mean) - 1.96*apply(trial_veteran$IBS, 2, sd)/sqrt(20)), 3), 
  "CI upper" = round(c(apply(trial_veteran$IBS, 2, mean) + 1.96*apply(trial_veteran$IBS, 2, sd)/sqrt(20)), 3)
  
)

#Experiment for prca --------------------------------------------------------------------

trial_prca = REAL_TRIAL(data = prca1, best_nhid = best_para_prca$best_nhid, best_nbase = best_para_prca$best_nbase,
                        nite = 20, threshold = 2, seed = 20)

###C-index
prca_C_index = data.frame(
  'Model' = c('BJ-ELM', 'BJ', 'BJ-LS', 'BJ-Trees', 'Coxph', 'RSF'), 
  'means' = round(c(apply(trial_prca$C_index, 2, mean)), 3), 
  "CI lower" = round(c(apply(trial_prca$C_index, 2, mean) - 1.96*apply(trial_prca$C_index, 2, sd)/sqrt(20)), 3), 
  "CI upper" = round(c(apply(trial_prca$C_index, 2, mean) + 1.96*apply(trial_prca$C_index, 2, sd)/sqrt(20)), 3)
  
)

##IBS
prca_IBS = data.frame(
  'Model'=c('BJ-ELM', 'BJ', 'BJ-LS', 'BJ-Trees', 'Coxph', 'RSF'), 
  'means'  =  round(c(apply(trial_prca$IBS, 2, mean)), 3), 
  "CI lower" = round(c(apply(trial_prca$IBS, 2, mean) - 1.96*apply(trial_prca$IBS, 2, sd)/sqrt(20)), 3), 
  "CI upper" = round(c(apply(trial_prca$IBS, 2, mean) + 1.96*apply(trial_prca$IBS, 2, sd)/sqrt(20)), 3)
  
)

# Output the results of the experiments of real data applications. --------
##(1)Table 3
table3 = data.frame('Dataset' = c('Pbc', 'Lung', 'WPBC', 'StageC', 'Prca', 'Veteran'), 
                   'n' = c(nrow(pbc), nrow(lung1), nrow(wpbc), nrow(stagec), nrow(prca1), nrow(veteran)), 
                   'p' = c(ncol(pbc) - 2, ncol(lung1) - 2, ncol(wpbc) - 2, ncol(stagec) - 2, ncol(prca1) - 2, ncol(veteran) - 2), 
                   'Censoring rate' = c(round(cen_pbc), round(cen_lung), round(cen_wpbc), round(cen_stagec), round(cen_prca), round(cen_veteran)), 
                   'M' = c(nrow(pbc) - nrow(pbc[complete.cases(pbc), ]), nrow(lung1) - nrow(lung1[complete.cases(lung1), ]), 
                         nrow(wpbc) - nrow(wpbc[complete.cases(wpbc), ]), nrow(stagec) - nrow(stagec[complete.cases(stagec), ]), 
                         nrow(prca1) - nrow(prca1[complete.cases(prca1), ]), nrow(veteran) - nrow(veteran[complete.cases(veteran), ])
                         ), 
                  'Data source' = c('randomForestSRC', ' survival', 'TH.data', 'rpart', ' SubgrPlots', ' survival')
                   )
##(2)Table 4
table4 = list("Pbc-C_index" = pbc_C_index, 
              "Pbc-IBS" = pbc_IBS, 
              "Lung-C_index" = lung_C_index, 
              "Lung-IBS" = lung_IBS, 
              "Wpbc-C_index" = wpbc_C_index, 
              "Wpbc-IBS" = wpbc_IBS, 
              "StageC-C_index" = stagec_C_index, 
              "StageC-IBS" = stagec_IBS, 
              "Veteran-C_index" = veteran_C_index, 
              "Veteran-IBS" = veteran_IBS,
              "prca-C_index" = prca_C_index, 
              "prca-IBS" = prca_IBS)

write_xlsx(table3, "BJ_ELM_Code/RESULT/Real_Data_Applications(Table_3).xlsx")
write_xlsx(table4, "BJ_ELM_Code/RESULT/Real_Data_Applications(Table_4).xlsx")
save.image('BJ_ELM_Code/RESULT/Real_Data_Applications_Results.Rdata')


