from __future__ import division
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 24 12:31:31 2016

@author: markus
"""

import numpy as np
import gc
import pandas as pd
from sklearn.cross_validation import StratifiedKFold
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn import metrics



#get some data & split into training and validation set
numeric_cols = [ 'L1_S24_F1846', 'L3_S32_F3850', 'L1_S24_F1695', 'L1_S24_F1632',
                     'L3_S33_F3855', 'L1_S24_F1604',
                     'L3_S29_F3407', 'L3_S33_F3865',
                     'L3_S38_F3952', 'L1_S24_F1723',
                     ]

x = pd.read_csv('train_numeric.csv', usecols = numeric_cols).fillna(6666666).astype(np.float32)
#x_test = pd.read_csv('test_numeric.csv', usecols = numeric_cols).fillna(6666666).astype(np.float32)
Y = pd.read_csv('train_numeric.csv', usecols = ['Response'])['Response'].astype(np.int8)
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

#models
prior = np.sum(Y_dev) / (1.*len(Y_dev))
layers = [
            [
            RandomForestClassifier(n_estimators = 4, max_features = 0.95, criterion = 'entropy', max_depth = 2, class_weight = 'balanced', n_jobs = -1, random_state = None),
            RandomForestClassifier(n_estimators = 4, max_features = 0.95, criterion = 'entropy', max_depth = 7, class_weight = 'balanced', n_jobs = -1, random_state = None),
            xgb.XGBClassifier(colsample_bytree = 0.85, learning_rate = 0.1, min_child_weight = 2, n_estimators = 4, subsample = 0.7, max_depth = 10, base_score = prior, missing = 6666666),
            ],
            [
            RandomForestClassifier(n_estimators = 1, max_features = 0.95, criterion = 'entropy', max_depth = 5, class_weight = 'balanced', n_jobs = -1, random_state = None),
            xgb.XGBClassifier(colsample_bytree = 0.85, learning_rate = 0.1, min_child_weight = 2.0, n_estimators = 1, subsample = 0.7, max_depth = 10, base_score = prior, missing = 6666666),
            ],   
            [
            xgb.XGBClassifier(colsample_bytree = 0.85, learning_rate = 0.1, min_child_weight = 2.0, n_estimators = 5, subsample = 0.7, max_depth = 2, base_score = prior, missing = 6666666),
            ]   
        ]


class stacked_generalizer(object):
    def __init__(self, x_test, layers = [], CV = []):
        self.x_test = x_test
        self.layers = layers
        self.CV = CV
        
    def __fit_layer(self, Y_dev, x_dev, x_test, learners):
        #generate bins
        skf = list(self.CV)
        
        # Pre-allocate the data
        blend_train = np.zeros((x_dev.shape[0], len(learners))) # Number of training data x Number of classifiers
        blend_test = np.zeros((x_test.shape[0], len(learners))) # Number of testing data x Number of classifiers
        
        # For each classifier, we train the number of fold times (=len(skf))
        for j, learner in enumerate(learners):
            print '>> model', j+1, type(learner).__name__
            blend_test_j = np.zeros((x_test.shape[0], len(skf))) # Number of testing data x Number of folds , we will take the mean of the predictions later
            for i, (train_index, cv_index) in enumerate(skf):
                print '>>>> computing fold', i+1
                
                # This is the training and validation set
                x_train = x_dev.iloc[train_index]
                Y_train = Y_dev.iloc[train_index]
                x_cv = x_dev.iloc[cv_index]
                #Y_cv = Y_dev.iloc[cv_index] #could run some diagnostics on performance
                
                learner.fit(x_train, Y_train)
                
                # This output will be the basis for our blended classifier to train against,
                # which is also the output of our classifiers
                blend_train[cv_index, j] = learner.predict_proba(x_cv)[:,1]
                blend_test_j[:, i] = learner.predict_proba(x_test)[:,1]
            # Take the mean of the predictions of the cross validation set
            blend_test[:, j] = blend_test_j.mean(1)
        learner_names = [(type(learner).__name__+str(i)) for i, learner in enumerate(learners)]
        return pd.DataFrame(blend_train, columns = learner_names), pd.DataFrame(blend_test, columns = learner_names)
    
    def fit(self, Y_train, x_train):
        for i, layer in enumerate(self.layers):
            if i+1 == len(self.layers): #compute every layer except for last one
                break
            else:
                print '> layer', i+1
                layer_train, layer_test = self.__fit_layer(Y_train, x_train, self.x_test, self.layers[i])
                x_train, self.x_test = layer_train, layer_test
        self.x_fitted_train = x_train
        self.x_fitted_test = self.x_test
        self.Y_train = Y_dev
        return self
    
    def predict_proba(self, x_test):
        if len(layers[len(layers)-1]) == 1: #if last layer consist of exactly one learner....
            stacker = layers[len(layers)-1][0].fit(self.x_fitted_train, self.Y_train)
            pred = stacker.predict_proba(self.x_fitted_test)[:, 1]
        else: #...else compute avg of all learners in last layer
            self.x_fitted_dev, self.x_fitted_test = self.__fit_layer(self.Y_train, self.x_fitted_train, self.x_fitted_test, self.layers[len(layers)-1])
            pred = self.x_fitted_test.mean(axis = 1)
        return pred
        
   

meta = stacked_generalizer(x_test = x_test, layers = layers, CV = StratifiedKFold(Y_dev, 2))
#meta.fit_layer(Y_dev, x_dev, x_test, layers[0])
meta.fit(Y_dev, x_dev)
pred = meta.predict_proba(x_test)


score = metrics.roc_auc_score(Y_test, pred)
print metrics.roc_auc_score.__name__, ':', score
    
single = layers[2][0].fit(x_dev, Y_dev)
predi = single.predict_proba(x_test)[:, 1]

score = metrics.roc_auc_score(Y_test, predi)
print metrics.roc_auc_score.__name__, ':', score


    
    


