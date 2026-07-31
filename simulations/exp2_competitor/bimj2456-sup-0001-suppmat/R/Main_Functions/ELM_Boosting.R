################################################################################
#
#                                                                                     
#   Filename    :    ELM_Boosting.R    												  
#   Project     :    Article "Buckley-James Boosting Model based on Extreme
#                    Learning Machine and Random Survival Forests"                                                             
#   Authors     :    Jianfen Kong, Shuhong Zhang                                                              
#   Date        :    20/02/2023
#   Purpose     :    Functions allowing to fit ELM-based boosting model.
#   Input data files  :    ---                                                        
#   Output data files :    ---
#
#   R Version   :    R-4.1.3                                                                
#   Required R packages :  MASS_7.3-55 
#                          purrr_0.3.4
#                          emplik_1.2
#                          rms_6.2-0
#                          caret_6.0-90
#                          rootSolve_1.8.2.3
#                          scorecard_0.3.6
#                          dplyr_1.0.10
#
################################################################################


library('MASS')
library('purrr')
library('emplik')
library('rms')
library('caret')
library('rootSolve')
library('scorecard')
library('caret')
library('dplyr')

# 1.ELM model ------------------------------------------------------------

actfun = function(x){
  
  #Activation function(sigmoid function)
  
  y = 1 / (1 + exp(-1*x))
  return(y)
}

randomMatrix = function(nCols, nRows) {
  
  #Function to produce the weights of the hidden layer in ELM.
  
  #ARGUMENTS
  #nCols: Number of columns of the matrix.
  #nRows: Number of rows of the matrix.
  
  #RETURN
  #A weights matrix of the hidden layer in ELM.
  
  myMat = matrix(runif(nCols*nRows, min = -1,max = 1), ncol = nCols)
  myMat 
}

elmtrain.default = function(x, y, nhid) {
  #Function for training the ELM model.
  
  # ARGUMENTS
  #x: A matrix. The columns of the input matrix should be of type numeric.
  #y: A matrix. In case of regression the matrix should have n rows and 1 column. 
  #nhid: A numeric value specifying the number of hidden neurons in ELM. 
  
  #RETURN
  #inpweight: The weight matrix of hidden layer.
  #biashid: The bias of hidden layer.
  #outweight: The output weight vector.
  #nhid: The number of hidden nodes.
  #fitted.values: Estimated y from the training samples. 
  
  if(nhid < 1) stop("ERROR: number of hidden neurons must be >= 1")
  
  T = t(y)
  P = t(x)
  
  inpweight = randomMatrix(nrow(P), nhid)  ##w-U(-1,1)
  tempH = inpweight %*% P
  biashid = runif(nhid,min = -1,max = 1)
  biasMatrix = matrix(rep(biashid, ncol(P)), nrow=nhid, ncol=ncol(P), byrow = F) 
  
  tempH = tempH + biasMatrix
  H = 1 / (1 + exp(-1*tempH))
  outweight = ginv(t(H), tol = sqrt(.Machine$double.eps)) %*% t(T)
  Y = t(t(H) %*% outweight)
  fitted.values = t(Y)
  residuals = y - fitted.values
  model = list(inpweight = inpweight, biashid = biashid, outweight = outweight, nhid = nhid, fitted.values = fitted.values)
  return(model)
}


# 2. ELM-based boosting model----------------------------------------------------

ELM_boosting = function(x, y, nhid = 20, nbase = 25, step = 0.1){
  
  #Function for training the ELM-based boosting model.
  
  # ARGUMENTS
  #x: A matrix. The columns of the input matrix should be of type numeric.
  #y: A matrix. In case of regression the matrix should have n rows and 1 column. 
  #nhid: The number of hidden neurons of each ELM. Defaults to 20.
  #nbase: The number of base learners. Defaults to 25.
  #step: A numeric value rangeing from 0 to 1  to specify the learning rate. Defaults to 0.1.
  
  #RETURN
  #y_hat: Estimated y from the training samples. 
  #hid_weights: A list saving nbase matrixs, each matrix is the weight matrix of hidden layer of each base learner ELM.
  #bias: A matrix, each column saving the bias of hidden layer of each ELM.
  #nhid: The number of hidden neurons of each ELM.
  #output_weights: A matrix, each column saving the output weight of each ELM.
  #residuals: A matrix, each column saving the fitted residual of each iteration.
  
  bias = matrix(NA, ncol = nbase, nrow = nhid)
  hid_weights = vector('list', length = nbase)
  output_weights = matrix(NA, ncol = nbase, nrow = nhid)
  target = matrix(NA, ncol = nbase + 1, nrow = nrow(x))
  target[ , 1] = y
  fit_value = matrix(NA, ncol = nbase, nrow = nrow(x))
  
  for(k in 1:nbase){
    base_learner = elmtrain.default(x = x, y = target[, k], nhid = nhid)
    bias[, k] = c(base_learner$biashid)
    hid_weights[[k]] = t(base_learner$inpweight)
    output_weights[ , k] = c(base_learner$outweight)
    target[, k + 1] = c(target[, k]) - c(step*base_learner$fitted.values)
    fit_value[, k] = c(base_learner$fitted.values) 
  }
  y_hat = step*(apply(fit_value, 1, sum))
  out = list('y_hat' = y_hat, 'hid_weights' = hid_weights, 'bias' = bias,
             'output_weights' = output_weights, 'residuals' = target, 'step' = step)
  return(out)
}




