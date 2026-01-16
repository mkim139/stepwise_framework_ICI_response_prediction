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
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.feature_selection import VarianceThreshold, chi2, SelectKBest, f_regression
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, RidgeClassifier
loo = LeaveOneOut()
from utilities.ML import ML_hyperparameters   

import warnings
from sklearn.exceptions import ConvergenceWarning

with warnings.catch_warnings():
    warnings.filterwarnings('ignore', 'Solver terminated early.*') #1
    warnings.filterwarnings('ignore', category=ConvergenceWarning) #2


#recursive filter
def filter(exp_dic,responses,train_idx,test_idx,shuffle_idx,diff_thres=.02,sens_thres=.85,return_coef=False,return_auc=False,orthomodel ='NetBio',cncount=5):
        #leverage point identification
        MLS = ['LogisticRegression']
        ################################### layer1 senescence
        sene_model ='NetBio-senescence'
        # orthomodel ='NetBio'
        sene_pred = np.array([0]*len(train_idx))
        net1_filt = np.array([0]*len(train_idx))
        sene_test_pred = 0
        sene_prev_model = None
        sene_prev_thres = None
        X_train_sene, X_test_sene, y_train_sene, y_test_sene = exp_dic[sene_model][train_idx], exp_dic[sene_model][test_idx], responses[train_idx], responses[test_idx]
        X_train_sene2, X_test_sene2, y_train_sene2, y_test_sene2 = exp_dic[orthomodel][train_idx], exp_dic[orthomodel][test_idx], responses[train_idx], responses[test_idx]

        X_train_sene = X_train_sene[shuffle_idx]
        y_train_sene = y_train_sene[shuffle_idx]
        X_train_sene2 = X_train_sene2[shuffle_idx]
        y_train_sene2 = y_train_sene2[shuffle_idx]

        init = True
       
        while True:
            #senescent model training
            gcv_senes = []
            gcv_net1s = []
            scores = []
            for ml in MLS:
                model, param_grid = ML_hyperparameters(ml)
                gcv_sene = GridSearchCV(model, param_grid=param_grid, scoring='roc_auc', cv=cncount, n_jobs=5).fit(X_train_sene, y_train_sene)
                
                gcv_senes.append(gcv_sene.best_estimator_)
                scores.append(gcv_sene.best_score_)
            sene_best_model = gcv_senes[np.argmax(scores)]
            sene_best_model_coef = sene_best_model.coef_
            sene_proba = cross_val_predict(sene_best_model, X_train_sene, y_train_sene, cv=cncount, method="predict_proba")[:,1]
            
            scores = []
            for ml in MLS:
                model, param_grid = ML_hyperparameters(ml)
                gcv_net1 = GridSearchCV(model, param_grid=param_grid, scoring='roc_auc', cv=cncount, n_jobs=5).fit(X_train_sene2[sene_pred==0], y_train_sene2[sene_pred==0])
                gcv_net1s.append(gcv_net1.best_estimator_)
                gcv_net1_coef = gcv_net1.best_estimator_.coef_
                scores.append(gcv_net1.best_score_)
            net1_best_model = gcv_net1s[np.argmax(scores)]
            net1_proba = cross_val_predict(net1_best_model, X_train_sene2[sene_pred==0], y_train_sene2[sene_pred==0], cv=cncount, method="predict_proba")[:,1]
            
            if sum(sene_pred)!=0:
                estimators = cross_validate(net1_best_model, X_train_sene2[sene_pred==0], y_train_sene2[sene_pred==0], cv=cncount, scoring='roc_auc', return_estimator=True)['estimator']
                restprobs = []
                for est in estimators:
                    restprobs.append(est.predict_proba(X_train_sene2[sene_pred==1])[:,1])
                restprobs = np.array(restprobs).T.mean(1)
                tprobs = np.zeros(len(y_train_sene2))
                tprobs[sene_pred==0] = net1_proba
                tprobs[sene_pred==1] = restprobs
                net1_proba = tprobs
            if return_auc & init:
                init_pred_prob = copy.deepcopy(net1_proba)
                init = False
            
            #### get best threshold for senescent model
            for thres in np.arange(0,1,0.005):
                y_pred = [1 if x > thres else 0 for x in sene_proba]
                neg_pred_rate = np.mean(np.array(y_train_sene)[np.array(y_pred)==0]==0)
                spec = np.mean(np.array(y_pred)[y_train_sene==0]==0)
                sens = np.mean(np.array(y_pred)[y_train_sene==1]==1)
                # if sens < .90:
                if sens < sens_thres:
                    break
            sene_thres = thres

            for thres in np.arange(0,1,0.005):
                y_pred = [1 if x > thres else 0 for x in net1_proba]
                neg_pred_rate = np.mean(np.array(y_train_sene2)[np.array(y_pred)==0]==0)
                spec = np.mean(np.array(y_pred)[y_train_sene2==0]==0)
                sens = np.mean(np.array(y_pred)[y_train_sene2==1]==1)
                if sens < .85:
                    break
            net1_thres = thres

            # diff_thres = .02

            new_sene_pred = ((net1_proba-sene_proba) > diff_thres) & (sene_proba < sene_thres)
            new_net1_filt = ((sene_proba-net1_proba) > diff_thres) & (net1_proba < net1_thres)

            temp_sene_pred = (sene_pred+new_sene_pred) > 0
            temp_net1_filt = (net1_filt+new_net1_filt) > 0

            if (np.mean(temp_sene_pred == sene_pred)==1):
                if sene_prev_model is not None:
                    sene_best_model = sene_prev_model
                    sene_thres = sene_prev_thres
                break
                #end while loop
            else:
                sene_pred = temp_sene_pred
                net1_filt = temp_net1_filt
                sene_prev_model = sene_best_model
                sene_prev_thres = sene_thres

        sene_proba = sene_best_model.predict_proba(X_test_sene)[0][1]
        net1_proba_test = net1_best_model.predict_proba(X_test_sene2)[0][1]

        sene_test_pred = ((net1_proba_test-sene_proba) > diff_thres) & (sene_proba < sene_thres) #sene_test_pred > 0 #

        if return_auc:
            filtered_init_pred = np.array(init_pred_prob)[sene_pred==0]
            filtered_trained_pred = np.array(net1_proba)[sene_pred==0]
            filtered_label = np.array(y_train_sene2)[sene_pred==0]
            #auc calc auprc, auroc
            trained_auprc = average_precision_score(filtered_label, filtered_trained_pred)
            trained_auroc = roc_auc_score(filtered_label, filtered_trained_pred)
            init_auprc = average_precision_score(filtered_label, filtered_init_pred)
            init_auroc = roc_auc_score(filtered_label, filtered_init_pred)
            aucs = {}
            aucs['trained_auprc'] = trained_auprc
            aucs['trained_auroc'] = trained_auroc
            aucs['init_auprc'] = init_auprc
            aucs['init_auroc'] = init_auroc
        
        train_sene_proba = sene_best_model.predict_proba(X_train_sene)[:,1]
        
        if not return_coef:
            if return_auc:
                return sene_pred, sene_proba, sene_test_pred, aucs , train_sene_proba
            else:
                return sene_pred, sene_proba, sene_test_pred, train_sene_proba
        else:
            return sene_pred, sene_proba, sene_test_pred, train_sene_proba, sene_best_model.coef_


def filter_fibro(exp_dic,responses,train_idx,test_idx,shuffle_idx,diff_thres=.02,sens_thres=.85,return_coef=False,return_auc=False):    
        cncount=5
        #leverage point identification
        MLS = ['LogisticRegression']
        ################################### layer1 senescence
        sene_model ='NetBio-fibroblast'
        # sene_model = 'collagen'
        orthomodel ='NetBio'
        sene_pred = np.array([0]*len(train_idx))
        net1_filt = np.array([0]*len(train_idx))
        sene_test_pred = 0
        sene_prev_model = None
        sene_prev_thres = None
        X_train_sene, X_test_sene, y_train_sene, y_test_sene = exp_dic[sene_model][train_idx], exp_dic[sene_model][test_idx], responses[train_idx], responses[test_idx]
        X_train_sene2, X_test_sene2, y_train_sene2, y_test_sene2 = exp_dic[orthomodel][train_idx], exp_dic[orthomodel][test_idx], responses[train_idx], responses[test_idx]

        X_train_sene = X_train_sene[shuffle_idx]
        y_train_sene = y_train_sene[shuffle_idx]
        X_train_sene2 = X_train_sene2[shuffle_idx]
        y_train_sene2 = y_train_sene2[shuffle_idx]

        init = True
       
        while True:
            #senescent model training
            gcv_senes = []
            gcv_net1s = []
            scores = []
            for ml in MLS:
                model, param_grid = ML_hyperparameters(ml)
                gcv_sene = GridSearchCV(model, param_grid=param_grid, scoring='roc_auc', cv=cncount, n_jobs=5).fit(X_train_sene, y_train_sene)
                
                gcv_senes.append(gcv_sene.best_estimator_)
                scores.append(gcv_sene.best_score_)
            sene_best_model = gcv_senes[np.argmax(scores)]
            sene_best_model_coef = sene_best_model.coef_
            sene_proba = cross_val_predict(sene_best_model, X_train_sene, y_train_sene, cv=cncount, method="predict_proba")[:,1]
            
            scores = []
            for ml in MLS:
                model, param_grid = ML_hyperparameters(ml)
                gcv_net1 = GridSearchCV(model, param_grid=param_grid, scoring='roc_auc', cv=cncount, n_jobs=5).fit(X_train_sene2[sene_pred==0], y_train_sene2[sene_pred==0])
                gcv_net1s.append(gcv_net1.best_estimator_)
                gcv_net1_coef = gcv_net1.best_estimator_.coef_
                scores.append(gcv_net1.best_score_)
            net1_best_model = gcv_net1s[np.argmax(scores)]
            net1_proba = cross_val_predict(net1_best_model, X_train_sene2[sene_pred==0], y_train_sene2[sene_pred==0], cv=cncount, method="predict_proba")[:,1]
            if sum(sene_pred)!=0:
                estimators = cross_validate(net1_best_model, X_train_sene2[sene_pred==0], y_train_sene2[sene_pred==0], cv=cncount, scoring='roc_auc', return_estimator=True)['estimator']
                restprobs = []
                for est in estimators:
                    restprobs.append(est.predict_proba(X_train_sene2[sene_pred==1])[:,1])
                restprobs = np.array(restprobs).T.mean(1)
                tprobs = np.zeros(len(y_train_sene2))
                tprobs[sene_pred==0] = net1_proba
                tprobs[sene_pred==1] = restprobs
                net1_proba = tprobs
            if return_auc & init:
                init_pred_prob = copy.deepcopy(net1_proba)
                init = False
            #### get best threshold for senescent model
            for thres in np.arange(0,1,0.005):
                y_pred = [1 if x > thres else 0 for x in sene_proba]
                neg_pred_rate = np.mean(np.array(y_train_sene)[np.array(y_pred)==0]==0)
                spec = np.mean(np.array(y_pred)[y_train_sene==0]==0)
                sens = np.mean(np.array(y_pred)[y_train_sene==1]==1)
                # if sens < .90:
                if sens < sens_thres:
                    break
            sene_thres = thres

            for thres in np.arange(0,1,0.005):
                y_pred = [1 if x > thres else 0 for x in net1_proba]
                neg_pred_rate = np.mean(np.array(y_train_sene2)[np.array(y_pred)==0]==0)
                spec = np.mean(np.array(y_pred)[y_train_sene2==0]==0)
                sens = np.mean(np.array(y_pred)[y_train_sene2==1]==1)
                if sens < .85:
                    break
            net1_thres = thres

            # diff_thres = .02

            new_sene_pred = ((net1_proba-sene_proba) > diff_thres) & (sene_proba < sene_thres)
            new_net1_filt = ((sene_proba-net1_proba) > diff_thres) & (net1_proba < net1_thres)

            temp_sene_pred = (sene_pred+new_sene_pred) > 0
            temp_net1_filt = (net1_filt+new_net1_filt) > 0

            if (np.mean(temp_sene_pred == sene_pred)==1):
                if sene_prev_model is not None:
                    sene_best_model = sene_prev_model
                    sene_thres = sene_prev_thres
                break
                #end while loop
            else:
                sene_pred = temp_sene_pred
                net1_filt = temp_net1_filt
                sene_prev_model = sene_best_model
                sene_prev_thres = sene_thres
                
        sene_proba = sene_best_model.predict_proba(X_test_sene)[0][1]
        net1_proba_test = net1_best_model.predict_proba(X_test_sene2)[0][1]
        sene_test_pred = ((net1_proba_test-sene_proba) > diff_thres) & (sene_proba < sene_thres) #sene_test_pred > 0 #
        


        if return_auc:
            filtered_init_pred = np.array(init_pred_prob)[sene_pred==0]
            filtered_trained_pred = np.array(net1_proba)[sene_pred==0]
            filtered_label = np.array(y_train_sene2)[sene_pred==0]
            #auc calc auprc, auroc
            trained_auprc = average_precision_score(filtered_label, filtered_trained_pred)
            trained_auroc = roc_auc_score(filtered_label, filtered_trained_pred)
            init_auprc = average_precision_score(filtered_label, filtered_init_pred)
            init_auroc = roc_auc_score(filtered_label, filtered_init_pred)
            aucs = {}
            aucs['trained_auprc'] = trained_auprc
            aucs['trained_auroc'] = trained_auroc
            aucs['init_auprc'] = init_auprc
            aucs['init_auroc'] = init_auroc
        
        if not return_coef:
            if return_auc:
                return sene_pred, sene_proba, sene_test_pred, aucs
            else:
                return sene_pred, sene_proba, sene_test_pred
        else:
            return sene_pred, sene_proba, sene_test_pred, sene_best_model.coef_
        
def filter_keap(exp_dic,responses,train_idx,test_idx,shuffle_idx,diff_thres=.02,sens_thres=.85,return_coef=False,return_auc=False):    
        cncount=5
        #leverage point identification
        MLS = ['LogisticRegression']
        ################################### layer1 senescence
        sene_model ='NetBio-KEAP'
        orthomodel ='NetBio'
        sene_pred = np.array([0]*len(train_idx))
        net1_filt = np.array([0]*len(train_idx))
        sene_test_pred = 0
        sene_prev_model = None
        sene_prev_thres = None
        X_train_sene, X_test_sene, y_train_sene, y_test_sene = exp_dic[sene_model][train_idx], exp_dic[sene_model][test_idx], responses[train_idx], responses[test_idx]
        X_train_sene2, X_test_sene2, y_train_sene2, y_test_sene2 = exp_dic[orthomodel][train_idx], exp_dic[orthomodel][test_idx], responses[train_idx], responses[test_idx]

        X_train_sene = X_train_sene[shuffle_idx]
        y_train_sene = y_train_sene[shuffle_idx]
        X_train_sene2 = X_train_sene2[shuffle_idx]
        y_train_sene2 = y_train_sene2[shuffle_idx]

        init = True
       
        while True:
            #senescent model training
            gcv_senes = []
            gcv_net1s = []
            scores = []
            for ml in MLS:
                model, param_grid = ML_hyperparameters(ml)
                gcv_sene = GridSearchCV(model, param_grid=param_grid, scoring='roc_auc', cv=cncount, n_jobs=5).fit(X_train_sene, y_train_sene)
                
                gcv_senes.append(gcv_sene.best_estimator_)
                scores.append(gcv_sene.best_score_)
            sene_best_model = gcv_senes[np.argmax(scores)]
            sene_best_model_coef = sene_best_model.coef_
            sene_proba = cross_val_predict(sene_best_model, X_train_sene, y_train_sene, cv=cncount, method="predict_proba")[:,1]
            
            scores = []
            for ml in MLS:
                model, param_grid = ML_hyperparameters(ml)
                gcv_net1 = GridSearchCV(model, param_grid=param_grid, scoring='roc_auc', cv=cncount, n_jobs=5).fit(X_train_sene2[sene_pred==0], y_train_sene2[sene_pred==0])
                gcv_net1s.append(gcv_net1.best_estimator_)
                gcv_net1_coef = gcv_net1.best_estimator_.coef_
                scores.append(gcv_net1.best_score_)
            net1_best_model = gcv_net1s[np.argmax(scores)]
            net1_proba = cross_val_predict(net1_best_model, X_train_sene2[sene_pred==0], y_train_sene2[sene_pred==0], cv=cncount, method="predict_proba")[:,1]
            if sum(sene_pred)!=0:
                estimators = cross_validate(net1_best_model, X_train_sene2[sene_pred==0], y_train_sene2[sene_pred==0], cv=cncount, scoring='roc_auc', return_estimator=True)['estimator']
                restprobs = []
                for est in estimators:
                    restprobs.append(est.predict_proba(X_train_sene2[sene_pred==1])[:,1])
                restprobs = np.array(restprobs).T.mean(1)
                tprobs = np.zeros(len(y_train_sene2))
                tprobs[sene_pred==0] = net1_proba
                tprobs[sene_pred==1] = restprobs
                net1_proba = tprobs
            if return_auc & init:
                init_pred_prob = copy.deepcopy(net1_proba)
                init = False
            #### get best threshold for senescent model
            for thres in np.arange(0,1,0.005):
                y_pred = [1 if x > thres else 0 for x in sene_proba]
                neg_pred_rate = np.mean(np.array(y_train_sene)[np.array(y_pred)==0]==0)
                spec = np.mean(np.array(y_pred)[y_train_sene==0]==0)
                sens = np.mean(np.array(y_pred)[y_train_sene==1]==1)
                # if sens < .90:
                if sens < sens_thres:
                    break
            sene_thres = thres

            for thres in np.arange(0,1,0.005):
                y_pred = [1 if x > thres else 0 for x in net1_proba]
                neg_pred_rate = np.mean(np.array(y_train_sene2)[np.array(y_pred)==0]==0)
                spec = np.mean(np.array(y_pred)[y_train_sene2==0]==0)
                sens = np.mean(np.array(y_pred)[y_train_sene2==1]==1)
                if sens < .85:
                    break
            net1_thres = thres

            # diff_thres = .02

            new_sene_pred = ((net1_proba-sene_proba) > diff_thres) & (sene_proba < sene_thres)
            new_net1_filt = ((sene_proba-net1_proba) > diff_thres) & (net1_proba < net1_thres)

            temp_sene_pred = (sene_pred+new_sene_pred) > 0
            temp_net1_filt = (net1_filt+new_net1_filt) > 0

            if (np.mean(temp_sene_pred == sene_pred)==1):
                if sene_prev_model is not None:
                    sene_best_model = sene_prev_model
                    sene_thres = sene_prev_thres
                break
                #end while loop
            else:
                sene_pred = temp_sene_pred
                net1_filt = temp_net1_filt
                sene_prev_model = sene_best_model
                sene_prev_thres = sene_thres
                
        sene_proba = sene_best_model.predict_proba(X_test_sene)[0][1]
        net1_proba_test = net1_best_model.predict_proba(X_test_sene2)[0][1]
        sene_test_pred = ((net1_proba_test-sene_proba) > diff_thres) & (sene_proba < sene_thres) #sene_test_pred > 0 #
        


        if return_auc:
            filtered_init_pred = np.array(init_pred_prob)[sene_pred==0]
            filtered_trained_pred = np.array(net1_proba)[sene_pred==0]
            filtered_label = np.array(y_train_sene2)[sene_pred==0]
            #auc calc auprc, auroc
            trained_auprc = average_precision_score(filtered_label, filtered_trained_pred)
            trained_auroc = roc_auc_score(filtered_label, filtered_trained_pred)
            init_auprc = average_precision_score(filtered_label, filtered_init_pred)
            init_auroc = roc_auc_score(filtered_label, filtered_init_pred)
            aucs = {}
            aucs['trained_auprc'] = trained_auprc
            aucs['trained_auroc'] = trained_auroc
            aucs['init_auprc'] = init_auprc
            aucs['init_auroc'] = init_auroc
        
        if not return_coef:
            if return_auc:
                return sene_pred, sene_proba, sene_test_pred, aucs
            else:
                return sene_pred, sene_proba, sene_test_pred
        else:
            return sene_pred, sene_proba, sene_test_pred, sene_best_model.coef_


def filter_random(exp_dic,responses,train_idx,test_idx,shuffle_idx,diff_thres=.02,sens_thres=.85,return_coef=False,return_auc=False):    
        cncount=5
        #leverage point identification
        MLS = ['LogisticRegression']
        ################################### layer1 senescence
        sene_model ='NetBio-senescence'
        orthomodel ='NetBio'
        sene_pred = np.array([0]*len(train_idx))
        net1_filt = np.array([0]*len(train_idx))
        sene_test_pred = 0
        sene_prev_model = None
        sene_prev_thres = None
        X_train_sene, X_test_sene, y_train_sene, y_test_sene = exp_dic[sene_model][train_idx], exp_dic[sene_model][test_idx], responses[train_idx], responses[test_idx]
        X_train_sene2, X_test_sene2, y_train_sene2, y_test_sene2 = exp_dic[orthomodel][train_idx], exp_dic[orthomodel][test_idx], responses[train_idx], responses[test_idx]

        X_train_sene = X_train_sene[shuffle_idx]
        y_train_sene = y_train_sene[shuffle_idx]
        X_train_sene2 = X_train_sene2[shuffle_idx]
        y_train_sene2 = y_train_sene2[shuffle_idx]

       
        while True:
            #senescent model training
            gcv_senes = []
            gcv_net1s = []
            scores = []
            for ml in MLS:
                model, param_grid = ML_hyperparameters(ml)
                gcv_sene = GridSearchCV(model, param_grid=param_grid, scoring='roc_auc', cv=cncount, n_jobs=5).fit(X_train_sene, y_train_sene)
                
                gcv_senes.append(gcv_sene.best_estimator_)
                scores.append(gcv_sene.best_score_)
            sene_best_model = gcv_senes[np.argmax(scores)]
            sene_best_model_coef = sene_best_model.coef_
            sene_proba = cross_val_predict(sene_best_model, X_train_sene, y_train_sene, cv=cncount, method="predict_proba")[:,1]
            
            scores = []
            for ml in MLS:
                model, param_grid = ML_hyperparameters(ml)
                gcv_net1 = GridSearchCV(model, param_grid=param_grid, scoring='roc_auc', cv=cncount, n_jobs=5).fit(X_train_sene2[sene_pred==0], y_train_sene2[sene_pred==0])
                gcv_net1s.append(gcv_net1.best_estimator_)
                gcv_net1_coef = gcv_net1.best_estimator_.coef_
                scores.append(gcv_net1.best_score_)
            net1_best_model = gcv_net1s[np.argmax(scores)]
            net1_proba = cross_val_predict(net1_best_model, X_train_sene2[sene_pred==0], y_train_sene2[sene_pred==0], cv=cncount, method="predict_proba")[:,1]
            if sum(sene_pred)!=0:
                estimators = cross_validate(net1_best_model, X_train_sene2[sene_pred==0], y_train_sene2[sene_pred==0], cv=cncount, scoring='roc_auc', return_estimator=True)['estimator']
                restprobs = []
                for est in estimators:
                    restprobs.append(est.predict_proba(X_train_sene2[sene_pred==1])[:,1])
                restprobs = np.array(restprobs).T.mean(1)
                tprobs = np.zeros(len(y_train_sene2))
                tprobs[sene_pred==0] = net1_proba
                tprobs[sene_pred==1] = restprobs
                net1_proba = tprobs

            #### get best threshold for senescent model
            for thres in np.arange(0,1,0.005):
                y_pred = [1 if x > thres else 0 for x in sene_proba]
                neg_pred_rate = np.mean(np.array(y_train_sene)[np.array(y_pred)==0]==0)
                spec = np.mean(np.array(y_pred)[y_train_sene==0]==0)
                sens = np.mean(np.array(y_pred)[y_train_sene==1]==1)
                # if sens < .90:
                if sens < sens_thres:
                    break
            sene_thres = thres

            for thres in np.arange(0,1,0.005):
                y_pred = [1 if x > thres else 0 for x in net1_proba]
                neg_pred_rate = np.mean(np.array(y_train_sene2)[np.array(y_pred)==0]==0)
                spec = np.mean(np.array(y_pred)[y_train_sene2==0]==0)
                sens = np.mean(np.array(y_pred)[y_train_sene2==1]==1)
                if sens < .85:
                    break
            net1_thres = thres

            # diff_thres = .02

            new_sene_pred = ((net1_proba-sene_proba) > diff_thres) & (sene_proba < sene_thres)
            new_net1_filt = ((sene_proba-net1_proba) > diff_thres) & (net1_proba < net1_thres)

            temp_sene_pred = (sene_pred+new_sene_pred) > 0
            temp_net1_filt = (net1_filt+new_net1_filt) > 0

            if (np.mean(temp_sene_pred == sene_pred)==1):
                if sene_prev_model is not None:
                    sene_best_model = sene_prev_model
                    sene_thres = sene_prev_thres
                break
                #end while loop
            else:
                sene_pred = temp_sene_pred
                net1_filt = temp_net1_filt
                sene_prev_model = sene_best_model
                sene_prev_thres = sene_thres
                
        sene_proba = sene_best_model.predict_proba(X_test_sene)[0][1]
        net1_proba_test = net1_best_model.predict_proba(X_test_sene2)[0][1]
        sene_test_pred = ((net1_proba_test-sene_proba) > diff_thres) & (sene_proba < sene_thres) #sene_test_pred > 0 #

        random_sene_pred = np.array([0]*len(y_train_sene2))
        resp_count = y_train_sene2[sene_pred==1].sum()
        nonresp_count = len(y_train_sene2[sene_pred==1])-resp_count
        if resp_count > 0:    
            random_resp_idx = np.random.choice(np.where(y_train_sene2==1)[0],resp_count,replace=False)
            random_sene_pred[random_resp_idx] = 1
        if nonresp_count > 0:
            random_nonresp_idx = np.random.choice(np.where(y_train_sene2==0)[0],nonresp_count,replace=False)
            random_sene_pred[random_nonresp_idx] = 1

        #retrain####################################################################################
        cncount=5
        #leverage point identification
        MLS = ['LogisticRegression']
        ################################### layer1 senescence
        sene_model ='NetBio-senescence'
        orthomodel ='NetBio'
        sene_pred = random_sene_pred
        net1_filt = np.array([0]*len(train_idx))
        sene_test_pred = 0
        sene_prev_model = None
        sene_prev_thres = None
        X_train_sene, X_test_sene, y_train_sene, y_test_sene = exp_dic[sene_model][train_idx], exp_dic[sene_model][test_idx], responses[train_idx], responses[test_idx]
        X_train_sene2, X_test_sene2, y_train_sene2, y_test_sene2 = exp_dic[orthomodel][train_idx], exp_dic[orthomodel][test_idx], responses[train_idx], responses[test_idx]

        X_train_sene = X_train_sene[shuffle_idx]
        y_train_sene = y_train_sene[shuffle_idx]
        X_train_sene2 = X_train_sene2[shuffle_idx]
        y_train_sene2 = y_train_sene2[shuffle_idx]

        gcv_senes = []
        gcv_net1s = []
        scores = []
        for ml in MLS:
            model, param_grid = ML_hyperparameters(ml)
            gcv_sene = GridSearchCV(model, param_grid=param_grid, scoring='roc_auc', cv=cncount, n_jobs=5).fit(X_train_sene, y_train_sene)
            
            gcv_senes.append(gcv_sene.best_estimator_)
            scores.append(gcv_sene.best_score_)
        sene_best_model = gcv_senes[np.argmax(scores)]
        sene_best_model_coef = sene_best_model.coef_
        sene_proba = cross_val_predict(sene_best_model, X_train_sene, y_train_sene, cv=cncount, method="predict_proba")[:,1]
        
        scores = []
        for ml in MLS:
            model, param_grid = ML_hyperparameters(ml)
            gcv_net1 = GridSearchCV(model, param_grid=param_grid, scoring='roc_auc', cv=cncount, n_jobs=5).fit(X_train_sene2[sene_pred==0], y_train_sene2[sene_pred==0])
            gcv_net1s.append(gcv_net1.best_estimator_)
            gcv_net1_coef = gcv_net1.best_estimator_.coef_
            scores.append(gcv_net1.best_score_)
        net1_best_model = gcv_net1s[np.argmax(scores)]
        net1_proba = cross_val_predict(net1_best_model, X_train_sene2[sene_pred==0], y_train_sene2[sene_pred==0], cv=cncount, method="predict_proba")[:,1]
        if sum(sene_pred)!=0:
            estimators = cross_validate(net1_best_model, X_train_sene2[sene_pred==0], y_train_sene2[sene_pred==0], cv=cncount, scoring='roc_auc', return_estimator=True)['estimator']
            restprobs = []
            for est in estimators:
                restprobs.append(est.predict_proba(X_train_sene2[sene_pred==1])[:,1])
            restprobs = np.array(restprobs).T.mean(1)
            tprobs = np.zeros(len(y_train_sene2))
            tprobs[sene_pred==0] = net1_proba
            tprobs[sene_pred==1] = restprobs
            net1_proba = tprobs

        #### get best threshold for senescent model
        for thres in np.arange(0,1,0.005):
            y_pred = [1 if x > thres else 0 for x in sene_proba]
            neg_pred_rate = np.mean(np.array(y_train_sene)[np.array(y_pred)==0]==0)
            spec = np.mean(np.array(y_pred)[y_train_sene==0]==0)
            sens = np.mean(np.array(y_pred)[y_train_sene==1]==1)
            # if sens < .90:
            if sens < sens_thres:
                break
        sene_thres = thres

        for thres in np.arange(0,1,0.005):
            y_pred = [1 if x > thres else 0 for x in net1_proba]
            neg_pred_rate = np.mean(np.array(y_train_sene2)[np.array(y_pred)==0]==0)
            spec = np.mean(np.array(y_pred)[y_train_sene2==0]==0)
            sens = np.mean(np.array(y_pred)[y_train_sene2==1]==1)
            if sens < .85:
                break
        net1_thres = thres

        new_sene_pred = ((net1_proba-sene_proba) > diff_thres) & (sene_proba < sene_thres)
        new_net1_filt = ((sene_proba-net1_proba) > diff_thres) & (net1_proba < net1_thres)

        temp_sene_pred = (sene_pred+new_sene_pred) > 0
        temp_net1_filt = (net1_filt+new_net1_filt) > 0

        sene_pred = temp_sene_pred
        net1_filt = temp_net1_filt

        sene_proba = sene_best_model.predict_proba(X_test_sene)[0][1]
        net1_proba_test = net1_best_model.predict_proba(X_test_sene2)[0][1]
        sene_test_pred = ((net1_proba_test-sene_proba) > diff_thres) & (sene_proba < sene_thres)

        return random_sene_pred, sene_proba, sene_test_pred

def filter_montecarlo(exp_dic,responses,train_idx,test_idx,diff_thres=.02,sens_thres=.85,cvcnt=5):
        return_auc = False    
        cncount=cvcnt
        #leverage point identification
        MLS = ['LogisticRegression']
        ################################### layer1 senescence
        sene_model ='NetBio-senescence'
        orthomodel ='NetBio'
        sene_pred = np.array([0]*len(train_idx))
        net1_filt = np.array([0]*len(train_idx))
        sene_test_pred = 0
        sene_prev_model = None
        sene_prev_thres = None
        X_train_sene, X_test_sene, y_train_sene, y_test_sene = exp_dic[sene_model][train_idx], exp_dic[sene_model][test_idx], responses[train_idx], responses[test_idx]
        X_train_sene2, X_test_sene2, y_train_sene2, y_test_sene2 = exp_dic[orthomodel][train_idx], exp_dic[orthomodel][test_idx], responses[train_idx], responses[test_idx]

        init = True
       
        while True:
            #senescent model training
            gcv_senes = []
            gcv_net1s = []
            scores = []
            for ml in MLS:
                model, param_grid = ML_hyperparameters(ml)
                gcv_sene = GridSearchCV(model, param_grid=param_grid, scoring='roc_auc', cv=cncount, n_jobs=5).fit(X_train_sene, y_train_sene)
                
                gcv_senes.append(gcv_sene.best_estimator_)
                scores.append(gcv_sene.best_score_)
            sene_best_model = gcv_senes[np.argmax(scores)]
            sene_best_model_coef = sene_best_model.coef_
            sene_proba = cross_val_predict(sene_best_model, X_train_sene, y_train_sene, cv=cncount, method="predict_proba")[:,1]
            
            scores = []
            for ml in MLS:
                model, param_grid = ML_hyperparameters(ml)
                gcv_net1 = GridSearchCV(model, param_grid=param_grid, scoring='roc_auc', cv=cncount, n_jobs=5).fit(X_train_sene2[sene_pred==0], y_train_sene2[sene_pred==0])
                gcv_net1s.append(gcv_net1.best_estimator_)
                gcv_net1_coef = gcv_net1.best_estimator_.coef_
                scores.append(gcv_net1.best_score_)
            net1_best_model = gcv_net1s[np.argmax(scores)]
            net1_proba = cross_val_predict(net1_best_model, X_train_sene2[sene_pred==0], y_train_sene2[sene_pred==0], cv=cncount, method="predict_proba")[:,1]
            if sum(sene_pred)!=0:
                estimators = cross_validate(net1_best_model, X_train_sene2[sene_pred==0], y_train_sene2[sene_pred==0], cv=cncount, scoring='roc_auc', return_estimator=True)['estimator']
                restprobs = []
                for est in estimators:
                    restprobs.append(est.predict_proba(X_train_sene2[sene_pred==1])[:,1])
                restprobs = np.array(restprobs).T.mean(1)
                tprobs = np.zeros(len(y_train_sene2))
                tprobs[sene_pred==0] = net1_proba
                tprobs[sene_pred==1] = restprobs
                net1_proba = tprobs
            if return_auc & init:
                init_pred_prob = copy.deepcopy(net1_proba)
                init = False
            #### get best threshold for senescent model
            for thres in np.arange(0,1,0.005):
                y_pred = [1 if x > thres else 0 for x in sene_proba]
                neg_pred_rate = np.mean(np.array(y_train_sene)[np.array(y_pred)==0]==0)
                spec = np.mean(np.array(y_pred)[y_train_sene==0]==0)
                sens = np.mean(np.array(y_pred)[y_train_sene==1]==1)
                # if sens < .90:
                if sens < sens_thres:
                    break
            sene_thres = thres

            for thres in np.arange(0,1,0.005):
                y_pred = [1 if x > thres else 0 for x in net1_proba]
                neg_pred_rate = np.mean(np.array(y_train_sene2)[np.array(y_pred)==0]==0)
                spec = np.mean(np.array(y_pred)[y_train_sene2==0]==0)
                sens = np.mean(np.array(y_pred)[y_train_sene2==1]==1)
                if sens < .85:
                    break
            net1_thres = thres

            # diff_thres = .02

            new_sene_pred = ((net1_proba-sene_proba) > diff_thres) & (sene_proba < sene_thres)
            new_net1_filt = ((sene_proba-net1_proba) > diff_thres) & (net1_proba < net1_thres)

            temp_sene_pred = (sene_pred+new_sene_pred) > 0
            temp_net1_filt = (net1_filt+new_net1_filt) > 0

            if (np.mean(temp_sene_pred == sene_pred)==1):
                if sene_prev_model is not None:
                    sene_best_model = sene_prev_model
                    sene_thres = sene_prev_thres
                break
                #end while loop
            else:
                sene_pred = temp_sene_pred
                net1_filt = temp_net1_filt
                sene_prev_model = sene_best_model
                sene_prev_thres = sene_thres
                
        sene_proba = sene_best_model.predict_proba(X_test_sene)[:,1]
        net1_proba_test = net1_best_model.predict_proba(X_test_sene2)[:,1]
        sene_test_pred = ((net1_proba_test-sene_proba) > diff_thres) & (sene_proba < sene_thres)

        return sene_pred, sene_proba, sene_test_pred
                        
def across_study_filter(train_dic, test_dic, y_train, y_test,shuffle_idx=None,ML='LogisticRegression',diff_thres=.02, sens_thres=.85,return_coef=False,return_proba=False):    
        cncount=5
        MLS = [ML]

        ################################### initialize
        sene_model ='NetBio-senescence'
        orthomodel ='NetBio'
        sene_prev_model = None
        sene_prev_thres = None


        X_train_sene, X_test_sene = train_dic[sene_model].T.values[1:], test_dic[sene_model].T.values[1:]
        X_train_sene2, X_test_sene2 = train_dic[orthomodel].T.values[1:], test_dic[orthomodel].T.values[1:]
        y_train_sene, y_train_sene2 = y_train, y_train
        y_test_sene, y_test_sene2 = y_test, y_test
        if shuffle_idx is not None:
            X_train_sene = X_train_sene[shuffle_idx]
            y_train_sene = y_train_sene[shuffle_idx]
            X_train_sene2 = X_train_sene2[shuffle_idx]
            y_train_sene2 = y_train_sene2[shuffle_idx]


        sene_pred = np.array([0]*len(y_train))
        net1_filt = np.array([0]*len(y_train))
        sene_test_pred = 0

        init = True
       
        while True:
            #senescent model training
            gcv_senes = []
            gcv_net1s = []
            scores = []
            for ml in MLS:
                if ml =='RandomForest':
                    model, param_grid = ML_hyperparameters(ml,X_train_sene)
                else:
                    model, param_grid = ML_hyperparameters(ml)
                gcv_sene = GridSearchCV(model, param_grid=param_grid, scoring='roc_auc', cv=cncount, n_jobs=5).fit(X_train_sene, y_train_sene)
                
                gcv_senes.append(gcv_sene.best_estimator_)
                scores.append(gcv_sene.best_score_)
            sene_best_model = gcv_senes[np.argmax(scores)]
            if ml == "LogisticRegression":
                sene_best_model_coef = sene_best_model.coef_
            else:
                sene_best_model_coef = None
            sene_proba = cross_val_predict(sene_best_model, X_train_sene, y_train_sene, cv=LeaveOneOut(), method="predict_proba")[:,1]
            
            scores = []
            for ml in MLS:
                if ml =='RandomForest':
                    model, param_grid = ML_hyperparameters(ml,X_train_sene2[sene_pred==0])
                else:
                    model, param_grid = ML_hyperparameters(ml)
                gcv_net1 = GridSearchCV(model, param_grid=param_grid, scoring='roc_auc', cv=cncount, n_jobs=5).fit(X_train_sene2[sene_pred==0], y_train_sene2[sene_pred==0])
                gcv_net1s.append(gcv_net1.best_estimator_)
                gcv_net1_coef = gcv_net1.best_estimator_.coef_
                scores.append(gcv_net1.best_score_)
            net1_best_model = gcv_net1s[np.argmax(scores)]
            net1_proba = cross_val_predict(net1_best_model, X_train_sene2[sene_pred==0], y_train_sene2[sene_pred==0], cv=LeaveOneOut(), method="predict_proba")[:,1]
            if sum(sene_pred)!=0:
                estimators = cross_validate(net1_best_model, X_train_sene2[sene_pred==0], y_train_sene2[sene_pred==0], cv=LeaveOneOut(), scoring='roc_auc', return_estimator=True)['estimator']
                restprobs = []
                for est in estimators:
                    restprobs.append(est.predict_proba(X_train_sene2[sene_pred==1])[:,1])
                restprobs = np.array(restprobs).T.mean(1)
                tprobs = np.zeros(len(y_train_sene2))
                tprobs[sene_pred==0] = net1_proba
                tprobs[sene_pred==1] = restprobs
                net1_proba = tprobs
            
            for thres in np.arange(0,1,0.005):
                import scipy.stats as stats
                fitnormalcdf = stats.norm.cdf(thres, loc=np.mean(sene_proba[y_train_sene==0]), \
                                              scale=np.std(sene_proba[y_train_sene==0]))
                fitnormalcdf1 = stats.norm.cdf(thres, loc=np.mean(sene_proba[y_train_sene==1]), \
                                              scale=np.std(sene_proba[y_train_sene==1]))
                TP = 1-fitnormalcdf1
                FP = 1-fitnormalcdf
                FN = fitnormalcdf1
                TN = fitnormalcdf
                sens = TP/(TP+FN)
                if sens < sens_thres:
                    break
            sene_thres = thres

            for thres in np.arange(0,1,0.005):
                y_pred = [1 if x > thres else 0 for x in net1_proba]
                neg_pred_rate = np.mean(np.array(y_train_sene2)[np.array(y_pred)==0]==0)
                spec = np.mean(np.array(y_pred)[y_train_sene2==0]==0)
                sens = np.mean(np.array(y_pred)[y_train_sene2==1]==1)
                if sens < .85:
                    break
            
            net1_thres = thres

            new_sene_pred = ((net1_proba-sene_proba) > diff_thres) & (sene_proba < sene_thres)

            # dist = (net1_proba[sene_proba < sene_thres]-sene_proba[sene_proba < sene_thres])
            # dist = dist[dist > 0]
            # seneauc = roc_auc_score(y_train_sene, sene_proba)
            # upper_bound = stats.norm.ppf((1-seneauc)/2,loc=np.mean(dist),scale=np.std(dist)) #.1
            # new_sene_pred = ((net1_proba-sene_proba) > upper_bound) & (sene_proba < sene_thres)

            

            new_net1_filt = ((sene_proba-net1_proba) > diff_thres) & (net1_proba < net1_thres)

            temp_sene_pred = (sene_pred+new_sene_pred) > 0
            temp_net1_filt = (net1_filt+new_net1_filt) > 0

            if (np.mean(temp_sene_pred == sene_pred)==1):
                if sene_prev_model is not None:
                    sene_best_model = sene_prev_model
                    sene_thres = sene_prev_thres
                break
                #end while loop
            else:
                sene_pred = temp_sene_pred
                net1_filt = temp_net1_filt
                sene_prev_model = sene_best_model
                sene_prev_thres = sene_thres

        train_sene_proba = sene_proba        
        sene_proba = sene_best_model.predict_proba(X_test_sene)[:,1] #[0][1]
        net1_proba_test = net1_best_model.predict_proba(X_test_sene2)[:,1] #[0][1]
        sene_test_pred = ((net1_proba_test-sene_proba) > diff_thres) & (sene_proba < sene_thres) #sene_test_pred > 0 #options
        if not return_proba:
            if return_coef:
                if ml == "LogisticRegression":
                    coef = net1_best_model.coef_[0]
                else:
                    coef = None
                return (net1_proba_test>.5)+0, net1_proba_test, sene_test_pred, coef, sene_pred #netbio pred, netbio proba, senescence pred coefnetbio
            else:
                return (net1_proba_test>.5)+0, net1_proba_test, sene_test_pred, sene_pred
        else:
            return (net1_proba_test>.5)+0, net1_proba_test, sene_test_pred, sene_pred, net1_proba, train_sene_proba,net1_proba_test,sene_proba  # train_net_proba, train_sene_proba,test_net_proba,test_sene_proba