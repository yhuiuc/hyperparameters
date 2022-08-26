#!/usr/bin/env python
# coding: utf-8

# Hyperparameter Selection via Global Sensitivity Analysis

# Author: Chirantan Ghosh and Yao Hu
# Date: 08/26/2022

print(__doc__)

import os

from sklearn.model_selection import cross_val_score

import pandas as pd
from matplotlib import style
style.use('fivethirtyeight')

# importing necessary libraries
import time
import calendar
from datetime import datetime
from pytz import timezone
from matplotlib import pyplot
import math
import numpy as np
import matplotlib.pyplot as plt
from statistics import mean, variance
import seaborn as sns
import xgboost as xgb

from sklearn import preprocessing
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.metrics import confusion_matrix

import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Sensitivity Analysis
from SALib.sample import saltelli
from SALib.analyze import sobol

import pickle

# predict the runoff occurrence and magnitude using XGBoost models
def prdRunoffXGBoost(file_path, datasets2, target, variables, param_values):

    # check if all variables are in datasets1 and datasets2
    colnames = datasets2.columns.to_list()
    if not all(col in colnames for col in variables):
        print('not all variables appear in the datasets 2')
        exit()
    
     #########################################################################################
    # Phase 1: training
    #########################################################################################
    # training data (70%) for the prediction of the occurrence of EOF events
    m = datasets2.shape[0]
    mt = int(m*0.7)
    
    # training data (70%) for the prediction of the magnitude of EOF events
    tr_m_xv = datasets2.loc[0:mt, variables]
    tr_m_yv_obs = datasets2.loc[0:mt, target]

    # validation data (30%) for the prediction of the magnitude of EOF events
    val_m_xv = datasets2.loc[mt:, variables]
    val_m_yv_obs = datasets2.loc[mt:, target]

    ########################################################################################
    # XGBoost Regressor
    ########################################################################################
    start = time.time()
    
    # list of hyperparameters
    l = param_values[:,0] # learning rate
    m = param_values[:,1] # max tree depth
    n = param_values[:,2] # min child weight
    u = param_values[:,3] # subsample rate
    e = param_values[:,4] # num of estimator
    c = param_values[:,5] # colsample by tree
    g = param_values[:,6] # gamma
    s = param_values[:,7] # seed
    r = param_values[:,8] # regularization lambda
    mae = []
    
    
    for i in range(0, 20): #4000
    # training the XGBoost models
        cv_reg_mod = xgb.XGBRegressor(
            learning_rate= l[i].astype(float).item(),
            max_depth= m[i].astype(int).item(),
            min_child_weight= n[i].astype(int).item(),
            subsample=u[i].astype(float).item(),
            n_estimators= e[i].astype(int).item(),
            colsample_bytree= c[i].astype(float).item(),
            gamma= g[i].astype(float).item(),
            seed= s[i].astype(int).item(),
            reg_lambda = r[i].astype(float).item(),
            objective='reg:squarederror')

        cv_reg_mod.fit(tr_m_xv, tr_m_yv_obs)
        tr_m_yv_mod = cv_reg_mod.predict(tr_m_xv)
        
        # calculate the variance
        # tr_m_mse = mean_squared_error(tr_m_yv_obs, tr_m_yv_mod)
        tr_m_mae = mean_absolute_error(tr_m_yv_obs, tr_m_yv_mod)
        tr_m_r2 = r2_score(tr_m_yv_obs, tr_m_yv_mod)
    
        print("MAE and R2 of Training: {0:.4f} and {1:.2f}".format(tr_m_mae, tr_m_r2))
        mae.append(tr_m_mae)
    
    end = time.time()
    print("Time for training the regression model: {0:.4f}s".format(end - start))
        
    return mae
    
    
########################################################################################
# Hyperparameter Selection
#########################################################################################

print('Hyperparameter Selection via Global Sensitivity Analysis:')

# read input data
current_path = os.getcwd()
file_path = current_path + '/data/Maumee/'
datasets2 = pd.read_csv(file_path + 'runoff.csv')


# Maumee, OH
# target variable, daily edge-of-field (EOF) runoff in Maumee, OH
target = ['RUNOFF']

# features used to predict
# 1. acsnom: daily accumulated melting water out of snow bottom the runoff [mm/day]
# 2. rainrate: daily precipitation [mm/day] 
# 3. sfhd: daily average depth of ponded water on the surface [mm/day]；sfhd_1: one-day lagged value of sfhd
# 4. fira: daily total net long-wave radiation to atmoshpere [mm/day];  
# 5. soil_t4: daily average soil temperature at the bottom [k/day]; soil_t4_1 and soil_t4_2: one- and two-day lagged values 
# 6. soil_m3: daily average volumetric soil moisture [-]; 
# 7. soilsat: daily soil integrated fraction of soil saturation [-]
# more information: 
# Hu, Y. et al., (2021). Edge-of-field runoff prediction by a hybrid modeling approach using causal inference. 
# Environmental Research Communications, 3(7), 075003.
variables = ['acsnom', 'rainrate', 'sfhd_1', 'sfhd', 'fira', 'soil_t4_1', 'soil_m3', 'soilsat_2', 'soil_t4_2']


# 'problem' is defined 
# hyperparameters: l: learning rate; m: max tree depth; n: min child weight; u: subsample rate; e: num of estimator; c: colsample by tree;
# g: gamma; s: seed; r: regularization lambda

problem = {'num_vars': 9,
           'names': ['l','m','n','u','e','c','g','s', 'r'],
           'bounds': [[0.0001, 0.1],
                     [1, 18],
                     [1, 18],
                     [0, 1],
                     [1000, 10000],
                     [0, 1],
                     [0, 1],
                     [1, 1000],
                     [0, 1]]
           }


# Generate samples
param_values = saltelli.sample(problem, 16) # sample size: 200
# print('sample size:{0}'.format(len(param_values)))

Y = prdRunoffXGBoost(file_path, datasets2, target, variables, param_values)

# The calucation of first and total order index
Mout = np.asarray(Y)

Si = sobol.analyze(problem, Mout, print_to_console=True)

# first-order index
print(Si['S1'])
# total-order index
print(Si['ST'])





