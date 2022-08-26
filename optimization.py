#!/usr/bin/env python
# coding: utf-8

# Hyperparameter Optimization via Bayesian Optimization
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
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.metrics import confusion_matrix

import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

import hyperopt
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials

import pickle
from xgboost import plot_tree
import matplotlib.pyplot as plt

from joblib import dump
from joblib import load
import warnings

warnings.filterwarnings("ignore")


TIMEFORMAT = '%m/%d/%y %H:%M'
TIMEFORMAT1 = '%m/%d/%y'


def prdRunoffXGBoost(datasets2, target, variables):

    colnames = datasets2.columns.to_list()
    if not all(col in colnames for col in variables):
        print('not all variables appear in the datasets 2')
        exit()
    
    #########################################################################################
    # Phase 1: training
    #########################################################################################
    
    # training data (70%) for the prediction of the occurrence of EOF events
    m = datasets2.shape[0]
    mt = int(m*0.70)

    tr_m_xv = datasets2.loc[0:mt, variables]
    tr_m_yv_obs = datasets2.loc[0:mt, target]

    
    # validation data (30%) for the prediction of the severity of EOF events
    val_m_xv = datasets2.loc[mt:, variables]
    val_m_yv_obs = datasets2.loc[mt:, target]
    
    space={'max_depth': hp.quniform("max_depth", 1, 18, 1), 
           'learning_rate': hp.loguniform('learning_rate', np.log(0.001), np.log(1)),
           'subsample': hp.uniform('subsample', 0.0, 1.0),
           'objective': 'reg:squarederror'}
            #  'gamma': hp.uniform ('gamma', 0.0, 1.0),
            #  'colsample_bytree' : hp.uniform('colsample_bytree', 0.0, 1.0),
            #  'min_child_weight' : hp.quniform('min_child_weight', 1, 15, 1),
            #  'n_estimators': hp.choice('n_estimators', [1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000]),
            #  'reg_lambda': hp.uniform('reg_lambda', 0.0, 1.0),
            #  'seed': hp.choice('seed', np.arange(1, 13+1, dtype=int)),
            #  'scale_pos_weight': hp.quniform('scale_pos_weight', 0, 15, 1)

                                                    
    # Regression:
    def hyperparameter_tuning(space):
        
        model = xgb.XGBRegressor(max_depth = int(space['max_depth']),
                                 subsample =space['subsample'], 
                                 learning_rate= space['learning_rate'], 
                                 objective= space['objective'])
                               
                               # gamma = space['gamma'],
                               # colsample_bytree =space['colsample_bytree'], 
                               # min_child_weight=space['min_child_weight'],
                               # n_estimators = space['n_estimators'], 
                               # reg_lambda = space['reg_lambda'], 
                               # seed = space['seed'],
                               # scale_pos_weight = space['scale_pos_weight'])
        
        evaluation = [( tr_m_xv, tr_m_yv_obs), ( val_m_xv, val_m_yv_obs)]
        

        model.fit(tr_m_xv, tr_m_yv_obs, eval_set=evaluation, eval_metric="mae", early_stopping_rounds=10,verbose=False)

        pred = model.predict(val_m_xv)
        
        mae = mean_absolute_error(val_m_yv_obs, pred)
        
            
        return {'loss':mae, 'status': STATUS_OK, 'model': model}
        
        # plot_tree(model)
        
        # plt.show()
        
    trials = Trials()
    
    optimals = fmin(fn=hyperparameter_tuning, space=space, algo=tpe.suggest, max_evals=1000, trials=trials)
    
    return optimals

    
#########################################################################################
# Hyperparameter Optimization
#########################################################################################

start = time.time()

print('Hyperparameter Optimization via Bayesian Optimization:')


# read input data
current_path = os.getcwd()
file_path = current_path + '/data/Maumee/'
datasets = pd.read_csv(file_path + 'runoff.csv')

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

# Optimial hyperparameters
opt_para = prdRunoffXGBoost(datasets, target, variables)

print (opt_para)

end = time.time()

# total time taken
print(f"Runtime of the program is {end - start}")





