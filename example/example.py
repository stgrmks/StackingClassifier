from __future__ import division
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 26 13:01:51 2016

@author: markus
"""
import sys
sys.path.append('..') # include higher directory to python modules path
import gc
import numpy as np
import pandas as pd
from sklearn.cross_validation import StratifiedKFold
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn import metrics
from multi_stacker import stacked_generalizer

# get some data & split into training and validation set
numeric_cols = [ 'L1_S24_F1846', 'L3_S32_F3850', 'L1_S24_F1695', 'L1_S24_F1632',
                     'L3_S33_F3855', 'L1_S24_F1604',
                     'L3_S29_F3407', 'L3_S33_F3865',
                     'L3_S38_F3952', 'L1_S24_F1723',
                     ]

x = pd.read_csv('train.csv.gz', usecols = numeric_cols, index_col = 0).fillna(6666666).astype(np.float32)
Y = pd.read_csv('Y.csv.gz', index_col = 0)['Response'].astype(np.int8)
shuffle_idx = np.random.permutation(Y.shape[0])
Y = Y.iloc[shuffle_idx]
x = x.iloc[shuffle_idx]
val_cutoff = int(len(Y) * 9/10)
x_dev = x.iloc[:val_cutoff]
Y_dev = Y.iloc[:val_cutoff]
x_test = x.iloc[val_cutoff:]
Y_test = Y.iloc[val_cutoff:]
del x, Y
gc.collect()

# models
prior = np.sum(Y_dev) / (1.*len(Y_dev))
layers = [
            [
            RandomForestClassifier(n_estimators = 2, max_features = 0.95, criterion = 'entropy', max_depth = 2, class_weight = 'balanced', n_jobs = -1, random_state = None),
            RandomForestClassifier(n_estimators = 2, max_features = 0.95, criterion = 'entropy', max_depth = 2, class_weight = 'balanced', n_jobs = -1, random_state = None),
            xgb.XGBClassifier(colsample_bytree = 0.85, learning_rate = 0.1, min_child_weight = 2, n_estimators = 2, subsample = 0.7, max_depth = 2, base_score = prior, missing = 6666666),
            RandomForestClassifier(n_estimators = 2, max_features = 0.95, criterion = 'entropy', max_depth = 2, class_weight = 'balanced', n_jobs = -1, random_state = None),
            xgb.XGBClassifier(colsample_bytree = 0.85, learning_rate = 0.1, min_child_weight = 2.0, n_estimators = 2, subsample = 0.7, max_depth = 2, base_score = prior, missing = 6666666),            
            ],
            [
            ExtraTreesClassifier(max_features = 0.95, n_estimators = 2, criterion = 'entropy', max_depth = 2, class_weight = 'balanced', n_jobs = -1, random_state = None),
            RandomForestClassifier(n_estimators = 2, max_features = 0.95, criterion = 'entropy', max_depth = 2, class_weight = 'balanced', n_jobs = -1, random_state = None),
            xgb.XGBClassifier(colsample_bytree = 0.85, learning_rate = 0.1, min_child_weight = 2, n_estimators = 2, subsample = 0.7, max_depth = 2, base_score = prior, missing = 6666666),
            RandomForestClassifier(n_estimators = 2, max_features = 0.95, criterion = 'entropy', max_depth = 2, class_weight = 'balanced', n_jobs = -1, random_state = None),
            xgb.XGBClassifier(colsample_bytree = 0.85, learning_rate = 0.1, min_child_weight = 2.0, n_estimators = 2, subsample = 0.7, max_depth = 2, base_score = prior, missing = 6666666),            
            ],   
            [
            ExtraTreesClassifier(max_features = 0.95, n_estimators = 2, criterion = 'entropy', max_depth = 2, class_weight = 'balanced', n_jobs = -1, random_state = None),
            xgb.XGBClassifier(colsample_bytree = 0.85, learning_rate = 0.1, min_child_weight = 2.0, n_estimators = 2, subsample = 0.7, max_depth = 2, base_score = prior, missing = 6666666),
            ]   
        ]

meta = stacked_generalizer(layers = layers, CV = StratifiedKFold(Y_dev, 2))
meta.fit(x_dev, Y_dev, x_test, keep_corr = True)
pred = meta.predict_proba()


score = metrics.roc_auc_score(Y_test, pred)
print metrics.roc_auc_score.__name__, ':', score

model_correlation_by_layer = meta.layer_corr()
 



