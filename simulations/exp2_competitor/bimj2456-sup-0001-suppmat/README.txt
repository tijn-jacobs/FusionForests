README

###############################################################################
Source code and data for the manuscript "Buckley-James Boosting Model based on Extreme
                                                               Learning Machine and Random Survival Forests" ,
Jianfen Kong, Shuhong Zhang.
###############################################################################

For questions, comments or remarks about the code please contact Shuhong Zhang, at Lanzhou University (shuhongzhang@lzu.edu.cn).

In our study, both simulation studies and real data applications are implemented using R-4.1.3 installed on a desktop computer operating on Windows 10 with 16 GB RAM and 8GB GPU. 
The system runs on an intel i5 processor with a maximum speed of 2.4 GHz.

R packages:
[1]purrr_0.3.4
[2]emplik_1.2
[3]Hmisc_4.6-0
[4]dplyr_1.0.10
[5]randomForestSRC_2.14.0
[6]pec_2021.10.11
[7]missForest_1.4
[8]SurvMetrics_0.5.0
[9]rms_6.2-0
[10]MASS_7.3-55
[11]caret_6.0-90
[12]rootSolve_1.8.2.3
[13]scorecard_0.3.6
[14]compareC_1.3.1 
[15]bujar_0.2-9
[16]gbm.2.1.8
[17]writexl_1.4.0
[18]compositions_2.0-4
[19]survival_3.2-13
[20]VIM_6.1.1

The main content and structure of the folder are as follows:

The subfolder 'R' contains all R code files to reproduce  the results presented in the manuscript ( Tables 1-4) . There are three subfolders in the folder 'R':
(1) Subfolder 'Main_Functions' contains three R documents: a)The R code 'BJ_ELM' is provided to fit the proposed ELM-based BJ boosting model; b)R code 'ELM_Boosting' is provided to 
      fit ELM-based  boosting model; c)R code 'Auxiliary_Functions' stores the support functions for implementing the experiments of real data applications and simulation studies.                  
(2) Subfolder 'Real_Data_Applications' is provided for reproducing the experimental results of six models on six clinical datasets (Table 3-4). In the subfolders, just run the file 'Real _Trials_Fast.R', 
     we can get the result showed in Table 3-4 in the manuscript. R code 'Real _Trials_Best_Parameters.R' is provided to determined the best Mstop(nbase) and the best number of hidden neurons(nhid)
     by 5-fold CV in 20 times experiments for each real survival dataset. The best nbase and the best nhid produced by running 'Real _Trials_Best_Parameters.R' will be stored in document 'Best_Para_For_Real_Trials.RData'
     as the intermediate results  to reduce the runtime when reproduce the results.
(3) Subfolder 'Simulation_Studies'  is provided for reproducing the experimental results of simulations (Table 1-2). Run the file 'Simulations.R', we can get the result showed in Table 1-2 in the manuscript.  

When running the R code 'Real _Trials_Fast.R' and 'Simulations.R' for reproduction, the experiment results contained in Table 1-4 will be stored in the subfolder 'RESULT' automatically .

/R
	./Main_Functions
                                  /Auxiliary_Functions.R
                                 /BJ_ELM.R
                                 /ELM_Boosting.R
	./Real_Data_Applications
                                /Real _Trials_Best_Parameters.R
                                /Real _Trials_Fast.R
                                /Best_Para_For_Real_Trials.RData
	./Simulation_Studies
                               /Simulations.R
                               /Simulate_Survival_Datasets.R                     

/RESULT







