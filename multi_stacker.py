from __future__ import division
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 24 12:31:31 2016

@author: markus
"""

import numpy as np
import pandas as pd
import matplotlib.pylab as plt
import seaborn as sns

class stacked_generalizer(object):
    def __init__(self, layers = [], CV = []):
        self.layers = layers
        self.CV = CV
        
    def __fit_layer_cv(self, Y_dev, x_dev, x_test, learners):
        # generate bins
        skf = list(self.CV)
        
        # pre-allocate the data
        blend_train = np.zeros((x_dev.shape[0], len(learners))) # number of training data x number of learners
        blend_test = np.zeros((x_test.shape[0], len(learners))) # number of testing data x number of learners
        
        # for each learner, we train the number of fold times (=len(skf))
        for j, learner in enumerate(learners):
            print '>> model', j+1, type(learner).__name__
            blend_test_j = np.zeros((x_test.shape[0], len(skf))) # number of testing data x number of folds , we will take the mean of the predictions later
            for i, (train_index, cv_index) in enumerate(skf):
                print '>>>> computing fold', i+1
                
                # this is the training and validation set
                x_train = x_dev.iloc[train_index]
                Y_train = Y_dev.iloc[train_index]
                x_cv = x_dev.iloc[cv_index]
                
                learner.fit(x_train, Y_train)
                
                # this output will be the basis for our meta learner to train against,
                # which is also the output of our learners
                blend_train[cv_index, j] = learner.predict_proba(x_cv)[:,1]
                blend_test_j[:, i] = learner.predict_proba(x_test)[:,1]
            # take the mean of the predictions of the cross validation set
            blend_test[:, j] = blend_test_j.mean(1) 
        learner_names = [(type(learner).__name__+str(i)) for i, learner in enumerate(learners)]
        return pd.DataFrame(blend_train, columns = learner_names), pd.DataFrame(blend_test, columns = learner_names)
    
    def __fit_layer(self, Y_dev, x_dev, x_test, learners):
        # pre-allocate the data
        blend_test = np.zeros((x_test.shape[0], len(learners))) # Number of testing data x number of learners
        
        # each learner will be trained and used for predicting test set
        for j, learner in enumerate(learners):
            print '>> model', j+1, type(learner).__name__
                   
            learner.fit(x_dev, Y_dev)
            
            # storing predictions
            blend_test[:, j] = learner.predict_proba(x_test)[:,1]

        learner_names = [(type(learner).__name__+str(i)) for i, learner in enumerate(learners)]
        return pd.DataFrame(blend_test, columns = learner_names)

    def fit(self, x_train, Y_train, x_test, keep_corr = False):
        self.corrs = None
        if keep_corr: 
            self.corrs = {}
        for i, layer in enumerate(self.layers):
            if i+1 == len(self.layers): # compute every layer except for last one
                break
            else:
                print '> layer', i+1
                layer_train, layer_test = self.__fit_layer_cv(Y_train, x_train, x_test, self.layers[i])
                if keep_corr: 
                    self.corrs['layer'+str(i+1)] = layer_train.corr()
                x_train, x_test = layer_train, layer_test
        self.x_fitted_train = x_train
        self.x_fitted_test = x_test
        self.Y_train = Y_train
    
    def predict_proba(self):
        print '> final layer'
        if len(self.layers[len(self.layers)-1]) == 1: # if last layer consist of exactly one learner....
            stacker = self.layers[len(self.layers)-1][0].fit(self.x_fitted_train, self.Y_train)
            print '>> stacker model', type(stacker).__name__
            pred = stacker.predict_proba(self.x_fitted_test)[:, 1]
        else: # ...else compute avg of all learners in last layer
            print '>> averaging over', [type(learner).__name__ for learner in self.layers[len(self.layers) - 1]]
            self.x_fitted_test = self.__fit_layer(self.Y_train, self.x_fitted_train, self.x_fitted_test, self.layers[len(self.layers)-1])
            pred = self.x_fitted_test.mean(axis = 1)
        return pred
    
    def layer_corr(self, size = 5, plot = False):
        if self.corrs is None:
            print 'correlations not stored during training!'
        else:
            if plot:
                for i, k in enumerate(self.corrs.keys()):
                    fig, ax = plt.subplots(figsize=(size, size))
                    sns.heatmap(self.corrs[k], xticklabels = self.corrs[k].columns.values, yticklabels = self.corrs[k].columns.values, ax = ax)
                    ax.set_title(k)
                    plt.show()
            return self.corrs
            
   



