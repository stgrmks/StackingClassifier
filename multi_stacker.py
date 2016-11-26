from __future__ import division
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 24 12:31:31 2016

@author: markus
"""

import numpy as np
import pandas as pd

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
        self.Y_train = Y_train
        return self
    
    def predict_proba(self, x_test):
        if len(self.layers[len(self.layers)-1]) == 1: #if last layer consist of exactly one learner....
            stacker = self.layers[len(self.layers)-1][0].fit(self.x_fitted_train, self.Y_train)
            pred = stacker.predict_proba(self.x_fitted_test)[:, 1]
        else: #...else compute avg of all learners in last layer
            self.x_fitted_dev, self.x_fitted_test = self.__fit_layer(self.Y_train, self.x_fitted_train, self.x_fitted_test, self.layers[len(self.layers)-1])
            pred = self.x_fitted_test.mean(axis = 1)
        return pred
        
   


    
    


