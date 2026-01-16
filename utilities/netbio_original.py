import pandas as pd
from collections import defaultdict
import numpy as np
import scipy.stats as stat
from statsmodels.stats.multitest import multipletests
import tqdm
import time, os, math, random
from scipy.stats import norm
import copy
import pickle

from sklearn.model_selection import cross_val_score, KFold, train_test_split, GridSearchCV, LeaveOneOut, cross_val_predict, cross_validate
from sklearn.metrics import *
from sklearn.feature_selection import VarianceThreshold, chi2, SelectKBest, f_regression
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.neural_network import MLPClassifier as mlp

import warnings
from sklearn.exceptions import ConvergenceWarning

loo = LeaveOneOut()

from utilities.ML import ML_hyperparameters


def netbio(X_train,y_train,sene_pred,mltype='LogisticRegression',cncount=5):
    # cncount=5
    gcvs = []
    scores = []
    MLS = [mltype]
    import warnings
    for ml in MLS:
        if ml == 'RandomForest':
            from sklearn.ensemble import RandomForestClassifier as RFC
            model = RFC()
            param_grid = {'n_estimators':[500, 1000], 'max_depth':[X_train.shape[0]], 'class_weight':['balanced']}
        elif ml == 'RandomForest2':
            from sklearn.ensemble import RandomForestClassifier as RFC
            model = RFC()
            param_grid = {'n_estimators':[1,5,10], 'max_depth':[X_train.shape[0]], 'class_weight':['balanced']}
        else:
            model, param_grid = ML_hyperparameters(ml)
        gcv = GridSearchCV(model, param_grid=param_grid, scoring='roc_auc', cv=cncount, n_jobs=5).fit(X_train[(sene_pred==0)], y_train[(sene_pred==0)])
        gcvs.append(gcv)
        scores.append(gcv.best_score_)
    ML = MLS[np.argmax(scores)]
    gcv = gcvs[np.argmax(scores)]
    return gcv


def netbio_mlp(X_train,y_train,sene_pred,cncount=5):
    import warnings
    import numpy as np
    from joblib import parallel_backend
    from sklearn.exceptions import ConvergenceWarning
    from sklearn.model_selection import GridSearchCV
    from sklearn.neural_network import MLPClassifier
    warnings.filterwarnings("ignore", category=ConvergenceWarning)
        
    cncount=cncount
    gcvs = []
    scores = []
    MLS = ['MLPClassifier']
    for ml in MLS:
        param_grid = {}
        param_grid['learning_rate_init'] = np.arange(0.001,.01,.002)
        param_grid['alpha'] = [0.0001, 0.001, 0.01]
        param_grid['hidden_layer_sizes'] = [(X_train.shape[0]//1,),(X_train.shape[0]//2,),(X_train.shape[0]//3,),(X_train.shape[0]//2, X_train.shape[0]//3,)]
        model = mlp(max_iter=500,learning_rate='invscaling')
        gcv = GridSearchCV(model, param_grid=param_grid, scoring='roc_auc', cv=cncount, n_jobs=1).fit(X_train[(sene_pred==0)], y_train[(sene_pred==0)])
        gcvs.append(gcv)
        scores.append(gcv.best_score_)
    ML = MLS[np.argmax(scores)]
    gcv = gcvs[np.argmax(scores)]
    return gcv