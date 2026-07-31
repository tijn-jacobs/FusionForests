################################################################################
#
#                                                                                     
#   Filename    :    real trial.R    												  
#   Project     :    Article "Buckley-James Boosting Model based on Extreme
#                    Learning Machine and Random Survival Forests"                                                             
#   Authors     :    Jianfen Kong, Shuhong Zhang                                                              
#   Date        :    15/12/2022
#   Purpose     :    Take the experiments of real data applications (Section 4.3 in the article.) 
#                      to produce table 4 which shows the results of real data applyications.
#   Input data files  :    ---                                                        
#   Output data files :    BJ_ELM_Code/R/Real_Data_Applications/Best_Para_For_Real_Trials.RData
#   R Version   :    R-4.1.3                                                              
#   Required R packages :  bujar_0.2-9, gbm.2.1.8 and writexl_1.4.0.
#
################################################################################


source('BJ_ELM_Code/R/Main_Functions/ELM_Boosting.R')
source('BJ_ELM_Code/R/Main_Functions/BJ_ELM.R')
source('BJ_ELM_Code/R/Main_Functions/Auxiliary_Functions.R')


REAL_TRIAL_PARA=function(data, nbase, nhid, nite, impute_BJ_ELM = 'no need', 
                    threshold = 5, seed = 2, skip = NULL){
                    
  #The function is developed to provide a overall framework for obtaining both the best Mstop and the best nhid 
  #chosen by k-fold CV in 20 times experiments.
  
  #ARGUMENTS
  #data: A dataframe with real survival data. The columns of survival time and censoring indicator
  #      should be named as 'time' and 'status' respectively.
  #nbase: Optional user-supplied Mstop sequence. Default to (10, 15, 20, 25, 30, 35).
  #nhid: Optional user-supplied sequence of the number of hidden neurons. Default to (5, 10, 15, 20, 25, 30).
  #nite: Number of times the experiment was repeated
  #seed: A numeric value specifying the random seed. Defaults to 1
  #skip: A vector contains the seeds that not all five competing models could training model successfully 
  #      under the data splitting based on the seed.
  
  
  #RETURN
  #best_nbase: A vector of the best Mstop chosen by k-fold CV in 20 times experiments.
  #best_nhid: A vector of the best nhid chosen by k-fold CV in 20 times experiments.

  data1 = data_trans(data, threshold = threshold)
  
  IBS_BJ_ELM = C_BJ_ELM = c()
  
  best_nbase = best_nhid = c()
  
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
    prepro_BJ_ELM = prepro_BJ_ELM(train, test, impute = impute_BJ_ELM, threshold = threshold)
    train_BJ_ELM = prepro_BJ_ELM$train_pro
    test_BJ_ELM = prepro_BJ_ELM$test_pro

    ##3.model

    ##(2)BJ_ELM
    BJ_ELM_start_time = Sys.time()
    cv_BJ_ELM = CV_BJ_ELM(data = train_BJ_ELM, k_folds = 5, nbase = nbase, nhid = nhid, seed = seed+i)
    best_nbase[i] = cv_BJ_ELM$best_nbase
    best_nhid[i] = cv_BJ_ELM$best_nhid
    
  }
  
  out = list( 'best_nbase' = best_nbase, 'best_nhid' = best_nhid) 

  return(out)
}


# Lung --------------------------------------------------------------------
##load the data
data(cancer, package = 'survival')
lung$status = lung$status-1
#On original lung data set, column "ph.ecog" is a discrete covariates which has 4 levels. On level '3', there are only 2 samples.
#These two samples may be assigned to the same data set (traning data or testing data), which lead to there is a column with variance 0 
#after one-hot ecoding this covariates. Therefore, in our study, we delete the two samples with level value '3' on covariates "ph.ecog".
lung1 = lung[which(lung$ph.ecog != 3), ]
cen_lung = 1 - sum(lung1$status)/nrow(lung1)
best_para_lung = REAL_TRIAL_PARA(data = lung1, nhid = seq(12, 27, by = 3), 
                        nbase  = seq(14, 29, by = 3), nite = 20, impute_BJ_ELM = 'RSF',  
                        threshold = 3, seed = 1, skip = c(5, 18))


# Pbc ---------------------------------------------------------------------
data(pbc, package = "randomForestSRC")
colnames(pbc)[which(colnames(pbc) == 'days')] = 'time'
cen_pbc = 1 - sum(pbc$status)/nrow(pbc)

best_para_pbc = REAL_TRIAL_PARA(data = pbc, nhid = seq(18, 36, by = 3), 
                       nbase  = seq(9, 23, by = 2), nite = 20, impute_BJ_ELM = 'RSF',  
                       threshold  =  3, seed  = 13)

#Experiment for WPBC --------------------------------------------------------------------
data("wpbc", package = 'TH.data')
wpbc$status = as.vector(wpbc$status)
wpbc$status[wpbc$status == 'R'] = 0
wpbc$status[wpbc$status == 'N'] = 1
wpbc$status = as.numeric(wpbc$status)
cen_wpbc = 1-sum(wpbc$status)/nrow(wpbc)

best_para_wpbc = REAL_TRIAL_PARA(data = wpbc, nhid = seq(8, 18, by = 2), nbase  = seq(12, 22, by = 2), nite = 20, 
                        impute_BJ_ELM = 'RSF', threshold  =  2, seed  = 29)


#Experiment for stageC ------------------------------------------------------------------

data(stagec, package = 'rpart')
colnames(stagec)[1:2] = c('time', 'status')
cen_stagec = 1-sum(stagec$status)/nrow(stagec)

best_para_stagec = REAL_TRIAL_PARA(data = stagec, nhid = seq(6, 20, by = 2), 
                                   nbase = seq(14, 28, by = 2), nite = 20, impute_BJ_ELM = 'RSF', 
                          threshold = 3, seed = 20)


#Experiment for veteran -----------------------------------------------------------------

data(cancer, package = 'survival')
cen_veteran = 1-sum(veteran$status)/nrow(veteran)
veteran$prior[which(veteran$prior==10)] = 1

best_para_veteran = REAL_TRIAL_PARA(data = veteran, nhid = seq(5, 20, by = 3), 
                                    nbase = seq(15, 27, by = 2), nite = 20, threshold = 4, seed = 51)


#Experiment for prca --------------------------------------------------------------------
data(prca, package = 'SubgrPlots')
colnames(prca)[which(colnames(prca)=='survtime')] = 'time'
colnames(prca)[which(colnames(prca)=='cens')] = 'status'
cen_prca = 1-sum(prca$status)/nrow(prca)
#On original prca data set, data of columns 10 to 15 is constructed based on the data of age (column 8) and weight (column 9),
#which means the information contained in the data in columns 10 to 15 has been included in the data in column 8 and column 9.
#Therefore, in our study, we delete data of the columns 10 to 15 on prca data set.
prca1 = prca[, 1:9]

best_para_prca = REAL_TRIAL_PARA(data = prca1, nhid = seq(6, 21, by = 3), 
                                 nbase  = seq(20, 35, by = 3), nite = 20, threshold = 2, seed = 20)

save.image("BJ_ELM_Code/R/Real_Data_Applications/Best_Para_For_Real_Trials.RData")

