__author__ = 'MSteger'

import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.base import ClassifierMixin

class stacker(ClassifierMixin):

    def __init__(self, layers = [], skf = None, verbose = 0):
        self.layers = layers
        self.skf = skf
        self.is_fitted = False
        self.classes_ = None
        self.verbose = verbose > 0

    def _fit_layer(self, X, y, models, average = False, last_layer = False):
        layer_pred = np.zeros((X.shape[0], len(self.classes_) - 1 + last_layer, len(models)))
        if not self.is_fitted:
            for i, (train_idx, test_idx) in enumerate(self.skf.split(X, y)):
                for j, model in enumerate(models):
                    if self.verbose:print 'bin {}: fitting model {}\n'.format(i, type(model).__name__)
                    model.fit(X[train_idx], y[train_idx])
                    bin_pred = model.predict_proba(X[test_idx])[:, 1:]
                    layer_pred[test_idx, :, j] = bin_pred
        else:
            for j, model in enumerate(models):
                if self.verbose: print 'predicting model {}\n'.format(type(model).__name__)
                bin_pred = model.predict_proba(X)[:, not last_layer:]
                layer_pred[:, :, j] = bin_pred
        if average or last_layer: layer_pred = layer_pred.mean(axis = -1)
        if len(layer_pred.shape) > 2: layer_pred = layer_pred.reshape(layer_pred.shape[0], reduce(lambda x, y: x*y, layer_pred.shape[1:]))
        return layer_pred

    def _iterate_layers(self, X, y = None):
        for i, models in enumerate(self.layers):
            if self.verbose: print 'layer {}:'.format(i)
            X = self._fit_layer(X = X, y = y, models = models, last_layer = (i == len(self.layers) - 1)*self.is_fitted)
        return X

    def fit(self, X, y):
        self.classes_ = np.unique(y)
        self._iterate_layers(X = X, y = y)
        self.is_fitted = True
        return self

    def predict(self, X):
        return self.predict_proba(X).argmax(axis = 1)

    def predict_proba(self, X):
        if not self.is_fitted: RuntimeError('You need to call fit() first!')
        X_pred = self._iterate_layers(X = X)
        return X_pred


if __name__ == '__main__':
    layers = [
        [
            RandomForestClassifier(n_jobs = -1),
            ExtraTreesClassifier(n_jobs = -1),
            RandomForestClassifier(n_jobs = -1),

        ],
        [
            RandomForestClassifier(n_jobs = 1),
            ExtraTreesClassifier(n_jobs = -1)
        ],
        [
            XGBClassifier(n_jobs = -1),
            ExtraTreesClassifier(n_jobs = 1)
        ]
    ]

    numeric_cols = ['L1_S24_F1846', 'L3_S32_F3850', 'L1_S24_F1695', 'L1_S24_F1632',
                    'L3_S33_F3855', 'L1_S24_F1604',
                    'L3_S29_F3407', 'L3_S33_F3865',
                    'L3_S38_F3952', 'L1_S24_F1723',
                    ]

    X = pd.read_csv('example/train.csv.gz', usecols=numeric_cols, index_col=0).fillna(6666666).astype(np.float32)
    y = pd.read_csv('example/Y.csv.gz', index_col=0)['Response'].astype(np.int8)

    ensemble = stacker(layers = layers, skf = StratifiedKFold(n_splits = 2, shuffle = True))
    ensemble.fit(X = X.as_matrix()[:5000], y = y.as_matrix()[:5000])
    yhat = ensemble.predict_proba(X.as_matrix()[5000:10000])
    from sklearn.metrics import accuracy_score
    print accuracy_score(y[5000:10000], yhat.argmax(axis = 1))

    print 'done'