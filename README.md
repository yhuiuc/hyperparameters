## HS-HT Version 1.0

### A methodological framework for improving the performance of data-driven models
*By Yao Hu, Chirantan Ghosh and Siamak Malakpour-Estalaki* 

The project aims to develop a framework for improving the efficacy of model training through two steps:

1. Hyperparameter Selection (sensitivity.py): identify the hyperparameters critical to the performance of the underlying ML algorithm of the data-driven model. In the case study, we used the data-driven model based on the eXtreme Gradient Boosting (XGBoost) algorithm to predict edge-of-field (EOF) runoff in the Maumee domain, US. 

2. Hyperparameter Tuning (optimization.py): search the optimal values for the chosen, influential hypaparameters. In our case, we identified three out of nine hyperparameters as influential hyperparameters.

As a result, we obtain an optimized XGBoost alogrithm that allows the data-driven model to be more efficiently and effectively to predict target system behaviors.

  
  
