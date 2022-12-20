#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 16 18:25:08 2020

@author: kevinarthurtittel
"""

import sys
sys.path.insert(0, "/Users/kevinarthurtittel/Downloads/assignment3")
import numpy as np
import pandas as pd
import datetime
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import accuracy_score
from MIRCO import MIRCO

# Import data and divide the attributes and outcome values and transform to numpy data

df = pd.read_csv('/Users/kevinarthurtittel/Downloads/assignment3/german.data-numeric.csv', header=None)
X = df.iloc[:, 0:-1]
y = df.iloc[:, -1]
X = X.to_numpy()
y = y.to_numpy()

# Setting up the parameter grids for the grid search
DT_paramgrid = {'max_depth': [5, 10, 20]}
    
RF_paramgrid = {'max_depth': [5, 10, 20],
                'n_estimators': [10, 50, 100]}

# Initialize decision tree and random forest classifiers 
DTclass = DecisionTreeClassifier(random_state=0)
RFclass = RandomForestClassifier(random_state=0)

performance = {'DT': [], 'RF': [], 'MIRCO': []}
num_rules = {'DT': [], 'RF': [], 'MIRCO': []}
missed_points_MIRCO = []

begin_time = datetime.datetime.now()

#Perform 10 x 3 -fold nested cross validation with stratification
stratified_k_fold = StratifiedKFold(n_splits=10, shuffle=True, random_state=0)
foldnumber = 0
for train_index, test_index in stratified_k_fold.split(X, y):
    foldnumber += 1
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    
    tuning=StratifiedKFold(n_splits=4, shuffle=True, random_state=0)
    for pgrid, est in zip((DT_paramgrid, RF_paramgrid), (DTclass, RFclass)):
        gridcv = GridSearchCV(estimator=est, param_grid=pgrid, scoring='accuracy', n_jobs=1, cv=tuning)
        gridcv_fit = gridcv.fit(X_train, y_train)
        
        # Use best estimator obtained from grid search
        gridcv_pred = gridcv_fit.best_estimator_.predict(X_test)
        if est == DTclass:
            performance['DT'].append(accuracy_score(gridcv_pred, y_test))
            num_rules['DT'].append(gridcv_fit.best_estimator_.tree_.n_leaves)
        else:
            performance['RF'].append(accuracy_score(gridcv_pred, y_test))
            MIRCO_estimator = MIRCO(gridcv_fit.best_estimator_)
            MIRCO_fit = MIRCO_estimator.fit(X_train, y_train)
            MIRCO_pred = MIRCO_fit.predict(X_test)
            performance['MIRCO'].append(accuracy_score(MIRCO_pred, y_test))
            num_rules['MIRCO'].append(MIRCO_fit.numOfRules)
            missed_points_MIRCO.append(MIRCO_fit.numOfMissed/len(y_test))
            num_rules['RF'].append(MIRCO_fit.initNumOfRules)
        
print(datetime.datetime.now() - begin_time)
        

