import pandas as pd
from collections import defaultdict
import numpy as np
import scipy.stats as stat
from statsmodels.stats.multitest import multipletests
import time, os, math, random
from scipy.stats import norm
import copy
import pickle

import warnings
warnings.filterwarnings("ignore")
import warnings
from sklearn.exceptions import ConvergenceWarning
warnings.filterwarnings("ignore", category=ConvergenceWarning)

import tqdm
import time
from utilities.netbio_original import netbio,netbio_mlp
from utilities.filter import filter as filtering
from utilities.filter import filter_fibro as filtering_fibro
from utilities.filter import filter_random

from sklearn.model_selection import cross_val_score, KFold, train_test_split, GridSearchCV, LeaveOneOut, cross_val_predict, cross_validate
from sklearn.metrics import *
from sklearn.feature_selection import VarianceThreshold, chi2, SelectKBest, f_regression
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, RidgeClassifier
loo = LeaveOneOut()

from utilities.useful_utilities import reactome_genes
from utilities.ML import ML_hyperparameters
from utilities.parse_patient_data import parse_reactomeExpression_and_immunotherapyResponse
import argparse

def repeated_loocv(filtertype,num_iters):

    auccomparetest = False

    ## Initialize options
    result_dir = './results'
    data_dir = './data/cohorts'
    biomarker_dir = './data/biomarker'

    patNum_cutoff = 1
    num_genes = 200
    qval_cutoff = 0.01
    cncount = 5


    num_iters = num_iters

    finalmltype = 'LogisticRegression'
    shuffle_traind = True #True


    ML = 'LogisticRegression'
    optimal = False

    ############################stepwise option##################
    '''filtertype = 'random', to randomly select patients to filter
    or filtertype = 'recursive', to use iterative stepwise framework to select patients to filter'''
    filtertype = filtertype #'recursive' #random
    #############################################################
    modeltesttype = 'netbio' 
    filtermodel = 'senescence' #'fibro'

    dt = 0.02
    st = .85
    testprop = .1

    predict_proba = False
    target_dic = {'Jung':'PD1_PD-L1', 'Liu':'PD1', 'Prat':'PD1', 
                'Gide':'PD1_CTLA4','Kim':'PD1','Mariathasan':'PD-L1',
                'PratNSCLC':'PD1'}
    datasets_to_test = target_dic.keys()
    datasets_to_test = ['PratNSCLC']


    # Reactome pathways
    reactome = reactome_genes()

    # biomarker genes
    bio_df = pd.read_csv('./data/Marker_summary_KEAP.txt', sep='\t')

    outputs = []
    pred_outputs = []
    stariter = 0
    init_iter = True
    for iter in range(stariter,num_iters):

        studydata = {}
        study_auccompares = {}
        study_rfcases = []

        ## output
        output = defaultdict(list)
        output_col = []
        proximity_df = defaultdict(list)

        output_two_step = defaultdict(list)
        output_two_step_col = []

        pred_output = defaultdict(list)
        pred_output_col = []

        output_svc = defaultdict(list)
        output_svc_col = []
        output_rf = defaultdict(list)
        output_rf_col = []
        output_mlp = defaultdict(list)
        output_mlp_col = []

        pred_output_svc = defaultdict(list)
        pred_output_svc_col = []
        pred_output_rf = defaultdict(list)
        pred_output_rf_col = []
        pred_output_mlp = defaultdict(list)
        pred_output_mlp_col = []
        

        pred_output_two_step = defaultdict(list)
        pred_output_two_step_col = []

        ## regularization parameter
        regularization_param = defaultdict(list)
        coefdata = {}
        aucdata = {}
        for fldr in os.listdir(data_dir):
            test_if_true = 0
            if ('et_al' in fldr) or ('IMvigor210' == fldr):
                if (fldr.replace('_et_al','') in datasets_to_test) or (fldr in datasets_to_test):
                    test_if_true = 1
                if test_if_true == 0:
                    continue
                print('')
                print('%s, %s'%(fldr, time.ctime()))
                study = fldr.split('_')[0]
                testTypes = []
                feature_name_dic = defaultdict(list) # { test_type : [ features ] }
                

                ## load immunotherapy datasets
                edf, epdf = pd.DataFrame(), pd.DataFrame()
                _, edf, epdf, responses = parse_reactomeExpression_and_immunotherapyResponse(study)
                
                # stop if number of patients are less than cutoff
                if edf.shape[1] < patNum_cutoff:
                    continue

                # scale (gene expression)
                tmp = StandardScaler().fit_transform(edf.T.values[1:])
                new_tmp = defaultdict(list)
                new_tmp['genes'] = edf['genes'].tolist()
                for s_idx, sample in enumerate(edf.columns[1:]):
                    new_tmp[sample] = tmp[s_idx]
                edf = pd.DataFrame(data=new_tmp, columns=edf.columns)

                # scale (pathway expression)
                tmp = StandardScaler().fit_transform(epdf.T.values[1:])
                new_tmp = defaultdict(list)
                new_tmp['pathway'] = epdf['pathway'].tolist()
                for s_idx, sample in enumerate(epdf.columns[1:]):
                    new_tmp[sample] = tmp[s_idx]
                epdf = pd.DataFrame(data=new_tmp, columns=epdf.columns)

            

                ## features, labels
                #exp_dic, responses = defaultdict(list), []
                exp_dic = defaultdict(list)
                sample_dic = defaultdict(list)


                ## load biomarker genes
                biomarkers = bio_df['Name'].tolist()
                bp_dic = defaultdict(list) # { biomarker : [ enriched pathways ] } // enriched functions
                tempdsets = {}
                for biomarker in biomarkers:
                    biomarker_genes = bio_df.loc[bio_df['Name']=='%s'%biomarker,:]['Gene_list'].tolist()[0].split(':')
                    exp_dic[biomarker] = edf.loc[edf['genes'].isin(biomarker_genes),:].T.values[1:]
                    sample_dic[biomarker] = edf.columns[1:]
                    feature_name_dic[biomarker] = edf.loc[edf['genes'].isin(biomarker_genes),:]['genes'].tolist()
                    testTypes.append(biomarker)
                    
                    
                    if '%s.txt'%biomarker in os.listdir(biomarker_dir):
                        # gene expansion by network propagation results
                        bdf = pd.read_csv('%s/%s.txt'%(biomarker_dir, biomarker), sep='\t')
                        bdf = bdf.dropna(subset=['gene_id'])
                        b_genes = []
                        for idx, gene in enumerate(bdf.sort_values(by=['propagate_score'], ascending=False)['gene_id'].tolist()):
                            if gene in edf['genes'].tolist():
                                if not gene in b_genes:
                                    b_genes.append(gene)
                                if len(set(b_genes)) >= num_genes:
                                    break
                        tmp_edf = edf.loc[edf['genes'].isin(b_genes),:]
                        
                        # enriched functions
                        tmp_hypergeom = defaultdict(list)
                        pvalues, qvalues = [], []; overlap = []; pw_count = []
                        for pw in list(reactome.keys()):
                            pw_genes = list(set(reactome[pw]) & set(edf['genes'].tolist()))
                            M = len(edf['genes'].tolist())
                            n = len(pw_genes)
                            N = len(set(b_genes))
                            k = len(set(pw_genes) & set(b_genes))
                            p = stat.hypergeom.sf(k-1, M, n, N)
                            tmp_hypergeom['pw'].append(pw)
                            tmp_hypergeom['p'].append(p)
                            pvalues.append(p)
                            overlap.append(k)
                            pw_count.append(n)
                            proximity_df['biomarker'].append(biomarker)
                            proximity_df['study'].append(study)
                            proximity_df['pw'].append(pw)
                            proximity_df['p'].append(p)
                        _, qvalues, _, _ = multipletests(pvalues)
                        for q in qvalues:
                            proximity_df['q'].append(q)
                        tmp_hypergeom['qval'] = qvalues
                        tmp_hypergeom['overlap'] = overlap
                        tmp_hypergeom['pw_count'] = pw_count
                        tmp_hypergeom = pd.DataFrame(tmp_hypergeom)
                        bp_dic[biomarker] = tmp_hypergeom.loc[tmp_hypergeom['qval']<=qval_cutoff,:]['pw'].tolist()
                        tmp_epdf = epdf.loc[epdf['pathway'].isin(bp_dic[biomarker]),:]
                        # NetBio
                        if target_dic[study] == biomarker:
                            exp_dic['NetBio'] = tmp_epdf.T.values[1:]
                            sample_dic['NetBio'] = tmp_epdf.columns[1:]
                            feature_name_dic['NetBio'] = tmp_epdf['pathway'].tolist()
                            testTypes.append('NetBio')
                            tempdsets['NetBio'] = copy.deepcopy(tmp_epdf)
                        if biomarker == 'senescence':
                            exp_dic['NetBio-senescence'] = tmp_epdf.T.values[1:]
                            sample_dic['NetBio-senescence'] = tmp_epdf.columns[1:]
                            feature_name_dic['NetBio-senescence'] = tmp_epdf['pathway'].tolist()
                            testTypes.append('NetBio-senescence')
                            tempdsets['NetBio-senescence'] = copy.deepcopy(tmp_epdf)
                        if (biomarker == 'KEAP'):
                            exp_dic['NetBio-KEAP'] = tmp_epdf.T.values[1:]
                            sample_dic['NetBio-KEAP'] = tmp_epdf.columns[1:]
                            feature_name_dic['NetBio-KEAP'] = tmp_epdf['pathway'].tolist()
                            testTypes.append('NetBio-KEAP')
                            tempdsets['NetBio-KEAP'] = copy.deepcopy(tmp_epdf)
                        if (biomarker == 'TGFB1'):
                            exp_dic['NetBio-fibroblast'] = tmp_epdf.T.values[1:]
                            sample_dic['NetBio-fibroblast'] = tmp_epdf.columns[1:]
                            feature_name_dic['NetBio-fibroblast'] = tmp_epdf['pathway'].tolist()
                            testTypes.append('NetBio-fibroblast')
                            tempdsets['NetBio-fibroblast'] = copy.deepcopy(tmp_epdf)
                        
                ##aggregate netbio, and netbio-senescence features
                feature_name_dic['Netbio+Netbio-senescence'] = list(set(tempdsets['NetBio']['pathway'].tolist()+tempdsets['NetBio-senescence']['pathway'].tolist()))
                newd = pd.concat([tempdsets['NetBio'],tempdsets['NetBio-senescence']],axis=0).drop_duplicates('pathway')
                exp_dic['Netbio+Netbio-senescence'] = newd.T.values[1:]
                sample_dic['Netbio+Netbio-senescence'] = newd.columns[1:]
                testTypes.append('Netbio+Netbio-senescence')

                ###aggregate netbio, netbio-fibro features
                feature_name_dic['Netbio+Netbio-fibroblast'] = list(set(tempdsets['NetBio']['pathway'].tolist()+tempdsets['NetBio-fibroblast']['pathway'].tolist()))
                newd = pd.concat([tempdsets['NetBio'],tempdsets['NetBio-fibroblast']],axis=0).drop_duplicates('pathway')
                exp_dic['Netbio+Netbio-fibroblast'] = newd.T.values[1:]
                sample_dic['Netbio+Netbio-fibroblast'] = newd.columns[1:]
                testTypes.append('Netbio+Netbio-fibroblast')

                feature_name_dic['Netbio+Netbio-KEAP'] = list(set(tempdsets['NetBio']['pathway'].tolist()+tempdsets['NetBio-KEAP']['pathway'].tolist()))
                newd = pd.concat([tempdsets['NetBio'],tempdsets['NetBio-KEAP']],axis=0).drop_duplicates('pathway')
                exp_dic['Netbio+Netbio-KEAP'] = newd.T.values[1:]
                sample_dic['Netbio+Netbio-KEAP'] = newd.columns[1:]
                testTypes.append('Netbio+Netbio-KEAP')
                
                ################################################################################ predictions
                
                ### LOOCV
                # if modeltesttype== 'other':
                # 	testTypes = [i for i in testTypes if ('PD' in i) or ('CTLA' in i)]
                # elif modeltesttype == 'netbio':

                testTypes = ['NetBio']

                selected_feature_dic = {}
                
                sf_counts =0
                for test_type in testTypes: # list(exp_dic.keys()):
                    if ('Prat'in study ) & (test_type=='CAF1'):
                        continue
                    obs_responses, pred_responses, pred_probabilities, pred_responses_two_step,pred_probabilities_two_step,obs_responses_two_step = [],[],[],[],[],[]
                    obs_responses_svc, pred_responses_svc, pred_probabilities_svc = [],[],[]
                    obs_responses_rf, pred_responses_rf, pred_probabilities_rf = [],[],[]
                    obs_responses_mlp, pred_responses_mlp, pred_probabilities_mlp = [],[],[]
                    selected_feature_dic[test_type] = defaultdict(list)

                    testmodelaucs,filterscores = [],[]
                    removesize = int(round(exp_dic[test_type].shape[0]*testprop,0))
                    removeidx = list(np.random.choice(exp_dic[test_type].shape[0],removesize,replace=False))
                    with tqdm.tqdm(total=exp_dic[test_type].shape[0]-removesize, desc="splits", unit="split") as pbar:
                        coefs = []
                        aucs_compares = []
                        for train_idx, test_idx in loo.split(exp_dic[test_type]):
                            if test_idx[0] in removeidx:
                                continue
                            train_idx = np.delete(train_idx, np.where(train_idx == test_idx[0]))
                            # train test split
                            X_train, X_test, y_train, y_test = exp_dic[test_type][train_idx], exp_dic[test_type][test_idx], responses[train_idx], responses[test_idx]
                            X_train_all, X_test_all, y_train_all, y_test_all = exp_dic['Netbio+Netbio-senescence'][train_idx], exp_dic['Netbio+Netbio-senescence'][test_idx], responses[train_idx], responses[test_idx]
                            orthomodel = 'NetBio'
                            done = None
                            if shuffle_traind:
                                random.seed(iter) #to control
                                shuffle_idx = np.arange(len(y_train))
                                if not init_iter:
                                    np.random.shuffle(shuffle_idx)
                            else:
                                shuffle_idx = np.arange(len(y_train))
                                
                            if filtertype == 'recursive':
                                if study == 'Prat':
                                    cnc = 3
                                else:
                                    cnc = 5
                                sene_pred, sene_proba, sene_test_pred, _ = filtering(exp_dic,responses,train_idx,test_idx,shuffle_idx,diff_thres=dt,sens_thres=st,return_auc=False,cncount=cnc)
                            elif (filtertype == 'random'):
                                sene_pred, sene_proba, sene_test_pred = filter_random(exp_dic,responses,train_idx,test_idx,shuffle_idx,diff_thres=dt,sens_thres=st)
                            if study == 'Prat':
                                cnc = cnc
                            else:
                                cnc = 5
                            gcv_single = netbio(X_train[shuffle_idx],y_train[shuffle_idx],np.array([0]*len(y_train)),mltype=finalmltype,cncount=cnc)
                            gcv_single_SVC = netbio(X_train_all[shuffle_idx],y_train[shuffle_idx],np.array([0]*len(y_train_all)),mltype='SVC')
                            gcv_single_rf = netbio(X_train_all[shuffle_idx],y_train[shuffle_idx],np.array([0]*len(y_train_all)),mltype='RandomForest')
                            gcv_single_mlp = netbio_mlp(X_train_all[shuffle_idx],y_train[shuffle_idx],np.array([0]*len(y_train_all)))
                            done = True
                            gcv_twostep = netbio(X_train[shuffle_idx],y_train[shuffle_idx],sene_pred,mltype=finalmltype)
                            trainsenepred = sene_pred
                
                            #############################stepwise result
                            # predictions
                            if (sene_test_pred==1):
                                pred_status = 0
                            elif predict_proba == False:
                                pred_status = gcv_twostep.best_estimator_.predict(X_test)[0]

                            if predict_proba:
                                pred_status = gcv_twostep.best_estimator_.predict_proba(X_test)[0][1]

                            obs_responses_two_step.append(y_test[0])
                            pred_responses_two_step.append(pred_status)
                            pred_probabilities_two_step.append(gcv_twostep.best_estimator_.predict_proba(X_test)[0][1])

                            # pred_output
                            pred_output_two_step['study'].append(study)
                            pred_output_two_step['test_type'].append(test_type)
                            pred_output_two_step['ML'].append(ML)
                            pred_output_two_step['nGene'].append(num_genes)
                            pred_output_two_step['qval'].append(qval_cutoff)
                            pred_output_two_step['sample'].append(sample_dic[test_type][test_idx[0]])
                            pred_output_two_step['predicted_response'].append(pred_status)
                            pred_output_two_step['pred_proba'].append(gcv_twostep.best_estimator_.predict_proba(X_test)[0][1])
                            pred_output_two_step['obs_response'].append(y_test[0])
                            pred_output_two_step['sene_proba'].append(sene_proba)
                            pred_output_two_step['sene_pred'].append(sene_test_pred)
                            sf_counts += sene_test_pred

                            ##########################original result
                            # predictions
                            if predict_proba :
                                pred_status = gcv_single.best_estimator_.predict_proba(X_test)[0][1]
                            else:
                                pred_status = gcv_single.best_estimator_.predict(X_test)[0]

                            obs_responses.append(y_test[0])
                            pred_responses.append(pred_status)
                            pred_probabilities.append(gcv_single.best_estimator_.predict_proba(X_test)[0][1])

                            # pred_output
                            pred_output['study'].append(study)
                            pred_output['test_type'].append(test_type)
                            pred_output['ML'].append(ML)
                            pred_output['nGene'].append(num_genes)
                            pred_output['qval'].append(qval_cutoff)
                            pred_output['sample'].append(sample_dic[test_type][test_idx[0]])
                            pred_output['predicted_response'].append(pred_status)
                            pred_output['pred_proba'].append(gcv_single.best_estimator_.predict_proba(X_test)[0][1])
                            pred_output['obs_response'].append(y_test[0])
                            ####################################################
                            ########################non-linear models
                            # svc
                            if predict_proba :
                                pred_status = gcv_single_SVC.best_estimator_.predict_proba(X_test_all)[0][1]
                            else:
                                pred_status = gcv_single_SVC.best_estimator_.predict(X_test_all)[0]
                            obs_responses_svc.append(y_test[0])
                            pred_responses_svc.append(pred_status)
                            pred_probabilities_svc.append(gcv_single_SVC.best_estimator_.predict_proba(X_test_all)[0][1])
                            # pred_output
                            pred_output_svc['study'].append(study)
                            pred_output_svc['test_type'].append(test_type)
                            pred_output_svc['ML'].append(ML)
                            pred_output_svc['nGene'].append(num_genes)
                            pred_output_svc['qval'].append(qval_cutoff)
                            pred_output_svc['sample'].append(sample_dic[test_type][test_idx[0]])
                            pred_output_svc['predicted_response'].append(pred_status)
                            pred_output_svc['pred_proba'].append(gcv_single_SVC.best_estimator_.predict_proba(X_test_all)[0][1])
                            pred_output_svc['obs_response'].append(y_test[0])
                            #rf
                            if predict_proba :
                                pred_status = gcv_single_rf.best_estimator_.predict_proba(X_test_all)[0][1]
                            else:
                                pred_status = gcv_single_rf.best_estimator_.predict(X_test_all)[0]
                            obs_responses_rf.append(y_test[0])
                            pred_responses_rf.append(pred_status)
                            pred_probabilities_rf.append(gcv_single_rf.best_estimator_.predict_proba(X_test_all)[0][1])
                            # pred_output
                            pred_output_rf['study'].append(study)
                            pred_output_rf['test_type'].append(test_type)
                            pred_output_rf['ML'].append(ML)
                            pred_output_rf['nGene'].append(num_genes)
                            pred_output_rf['qval'].append(qval_cutoff)
                            pred_output_rf['sample'].append(sample_dic[test_type][test_idx[0]])
                            pred_output_rf['predicted_response'].append(pred_status)
                            pred_output_rf['pred_proba'].append(gcv_single_rf.best_estimator_.predict_proba(X_test_all)[0][1])
                            pred_output_rf['obs_response'].append(y_test[0])
                            #mlp
                            if predict_proba :
                                pred_status = gcv_single_mlp.best_estimator_.predict_proba(X_test_all)[0][1]
                            else:
                                pred_status = gcv_single_mlp.best_estimator_.predict(X_test_all)[0]
                            obs_responses_mlp.append(y_test[0])
                            pred_responses_mlp.append(pred_status)
                            pred_probabilities_mlp.append(gcv_single_mlp.best_estimator_.predict_proba(X_test_all)[0][1])
                            # pred_output
                            pred_output_mlp['study'].append(study)
                            pred_output_mlp['test_type'].append(test_type)
                            pred_output_mlp['ML'].append(ML)
                            pred_output_mlp['nGene'].append(num_genes)
                            pred_output_mlp['qval'].append(qval_cutoff)
                            pred_output_mlp['sample'].append(sample_dic[test_type][test_idx[0]])
                            pred_output_mlp['predicted_response'].append(pred_status)
                            pred_output_mlp['pred_proba'].append(gcv_single_mlp.best_estimator_.predict_proba(X_test_all)[0][1])
                            pred_output_mlp['obs_response'].append(y_test[0])

                            pbar.update(1)

                    if len(pred_responses)==0:
                        continue
                    # predict dataframe
                    pdf = defaultdict(list)
                    pdf['obs'] = obs_responses
                    pdf['pred'] = pred_responses
                    pdf = pd.DataFrame(data=pdf, columns=['obs', 'pred'])
                    pdf_two_step = defaultdict(list)
                    pdf_two_step['obs'] = obs_responses_two_step
                    pdf_two_step['pred'] = pred_responses_two_step
                    pdf_two_step = pd.DataFrame(data=pdf_two_step, columns=['obs', 'pred'])
                    
                    ## output dataframe
                    for key, value in zip(['study', 'test_type', 'ML', 'nGene', 'qval'], [study, test_type, ML, num_genes, qval_cutoff]):
                        if key in ['nGene', 'qval']:
                            if 'NetBio' in test_type:
                                output[key].append(value)
                            else:
                                output[key].append('na')
                        else:
                            output[key].append(value)

                    for key, value in zip(['study', 'test_type', 'ML', 'nGene', 'qval'], [study, test_type, ML, num_genes, qval_cutoff]):
                        if key in ['nGene', 'qval']:
                            if 'NetBio' in test_type:
                                output_svc[key].append(value)
                            else:
                                output_svc[key].append('na')
                        else:
                            output_svc[key].append(value)
                    
                    for key, value in zip(['study', 'test_type', 'ML', 'nGene', 'qval'], [study, test_type, ML, num_genes, qval_cutoff]):
                        if key in ['nGene', 'qval']:
                            if 'NetBio' in test_type:
                                output_rf[key].append(value)
                            else:
                                output_rf[key].append('na')
                        else:
                            output_rf[key].append(value)
                    
                    for key, value in zip(['study', 'test_type', 'ML', 'nGene', 'qval'], [study, test_type, ML, num_genes, qval_cutoff]):
                        if key in ['nGene', 'qval']:
                            if 'NetBio' in test_type:
                                output_mlp[key].append(value)
                            else:
                                output_mlp[key].append('na')
                        else:
                            output_mlp[key].append(value)

                    for key, value in zip(['study', 'test_type', 'ML', 'nGene', 'qval'], [study, test_type, ML, num_genes, qval_cutoff]):
                        if key in ['nGene', 'qval']:
                            if 'NetBio' in test_type:
                                output_two_step[key].append(value)
                            else:
                                output_two_step[key].append('na')
                        else:
                            output_two_step[key].append(value)

                    # accuracy
                    accuracy = accuracy_score(obs_responses, pred_responses)
                    output['accuracy'].append(accuracy)
                    accuracy = accuracy_score(obs_responses, pred_responses_two_step)
                    output_two_step['accuracy'].append(accuracy)
                    accuracy= accuracy_score(obs_responses, pred_responses_rf)
                    output_rf['accuracy'].append(accuracy)
                    accuracy = accuracy_score(obs_responses, pred_responses_mlp)
                    output_mlp['accuracy'].append(accuracy)
                    accuracy = accuracy_score(obs_responses, pred_responses_svc)
                    output_svc['accuracy'].append(accuracy)

                    # precision
                    precision = precision_score(obs_responses, pred_responses, pos_label=1)
                    output['precision'].append(precision)
                    precision = precision_score(obs_responses, pred_responses_two_step, pos_label=1)
                    output_two_step['precision'].append(precision)
                    precision = precision_score(obs_responses, pred_responses_rf, pos_label=1)
                    output_rf['precision'].append(precision)
                    precision = precision_score(obs_responses, pred_responses_mlp, pos_label=1)
                    output_mlp['precision'].append(precision)
                    precision = precision_score(obs_responses, pred_responses_svc, pos_label=1)
                    output_svc['precision'].append(precision)

                    # recall 
                    recall = recall_score(obs_responses, pred_responses, pos_label=1)
                    output['recall'].append(recall)
                    recall = recall_score(obs_responses, pred_responses_two_step, pos_label=1)
                    output_two_step['recall'].append(recall)
                    recall = recall_score(obs_responses, pred_responses_rf, pos_label=1)
                    output_rf['recall'].append(recall)
                    recall = recall_score(obs_responses, pred_responses_mlp, pos_label=1)
                    output_mlp['recall'].append(recall)
                    recall = recall_score(obs_responses, pred_responses_svc, pos_label=1)
                    output_svc['recall'].append(recall)

                    # F1
                    F1 = f1_score(obs_responses, pred_responses, pos_label=1)
                    output['F1'].append(F1)
                    F1 = f1_score(obs_responses, pred_responses_two_step, pos_label=1)
                    output_two_step['F1'].append(F1)
                    F1 = f1_score(obs_responses, pred_responses_rf, pos_label=1)
                    output_rf['F1'].append(F1)
                    F1 = f1_score(obs_responses, pred_responses_mlp, pos_label=1)
                    output_mlp['F1'].append(F1)
                    F1 = f1_score(obs_responses, pred_responses_svc, pos_label=1)
                    output_svc['F1'].append(F1)

                    # auc
                    fpr, tpr, thresholds = roc_curve(obs_responses, pred_probabilities)
                    auc_val = auc(fpr, tpr)
                    output['AUC'].append(auc_val)
                    fpr, tpr, thresholds = roc_curve(obs_responses, pred_probabilities_two_step)
                    auc_val = auc(fpr, tpr)
                    output_two_step['AUC'].append(auc_val)
                    fpr, tpr, thresholds = roc_curve(obs_responses, pred_probabilities_rf)
                    auc_val = auc(fpr, tpr)
                    output_rf['AUC'].append(auc_val)
                    fpr, tpr, thresholds = roc_curve(obs_responses, pred_probabilities_mlp)
                    auc_val = auc(fpr, tpr)
                    output_mlp['AUC'].append(auc_val)
                    fpr, tpr, thresholds = roc_curve(obs_responses, pred_probabilities_svc)
                    auc_val = auc(fpr, tpr)
                    output_svc['AUC'].append(auc_val)

                    #filtered
                    # TP, TN, FP, FN, sensitivity, specificity
                    tn, fp, fn, tp = confusion_matrix(obs_responses, pred_responses).ravel()
                    sensitivity = tp/(tp+fn)
                    specificity = tn/(tn+fp)
                    for key, value in zip(['TP', 'TN', 'FP', 'FN', 'sensitivity', 'specificity'], [tp, tn, fp, fn, sensitivity, specificity]):
                        output[key].append(value)
                    tn, fp, fn, tp = confusion_matrix(obs_responses_two_step, pred_responses_two_step).ravel()
                    sensitivity = tp/(tp+fn)
                    specificity = tn/(tn+fp)
                    for key, value in zip(['TP', 'TN', 'FP', 'FN', 'sensitivity', 'specificity'], [tp, tn, fp, fn, sensitivity, specificity]):
                        output_two_step[key].append(value)
                    tn, fp, fn, tp = confusion_matrix(obs_responses_rf, pred_responses_rf).ravel()
                    sensitivity = tp/(tp+fn)
                    specificity = tn/(tn+fp)
                    for key, value in zip(['TP', 'TN', 'FP', 'FN', 'sensitivity', 'specificity'], [tp, tn, fp, fn, sensitivity, specificity]):
                        output_rf[key].append(value)
                    tn, fp, fn, tp = confusion_matrix(obs_responses_mlp, pred_responses_mlp).ravel()
                    sensitivity = tp/(tp+fn)
                    specificity = tn/(tn+fp)
                    for key, value in zip(['TP', 'TN', 'FP', 'FN', 'sensitivity', 'specificity'], [tp, tn, fp, fn, sensitivity, specificity]):
                        output_mlp[key].append(value)
                    tn, fp, fn, tp = confusion_matrix(obs_responses_svc, pred_responses_svc).ravel()
                    sensitivity = tp/(tp+fn)
                    specificity = tn/(tn+fp)
                    for key, value in zip(['TP', 'TN', 'FP', 'FN', 'sensitivity', 'specificity'], [tp, tn, fp, fn, sensitivity, specificity]):
                        output_svc[key].append(value)	


        output = pd.DataFrame(output)
        output['ML'] = 'TM'
        output_two_step = pd.DataFrame(output_two_step)
        output_two_step['ML'] = 'Stepwise'
        output_rf = pd.DataFrame(output_rf)
        output_rf['ML'] = 'ffRandomForest'
        output_mlp = pd.DataFrame(output_mlp)
        output_mlp['ML'] = 'ffMLP'
        output_svc = pd.DataFrame(output_svc)
        output_svc['ML'] = 'ffSVC'
        
        proximity_df = pd.DataFrame(proximity_df)

        outputs.append((output,output_two_step,output_rf,output_mlp,output_svc))
        pred_outputs.append((pred_output,pred_output_two_step,pred_output_rf,pred_output_mlp,pred_output_svc))
        outputs.append((output,output_two_step))
        pred_outputs.append((pred_output,pred_output_two_step))
        print('iteration %s done'%(iter+1))

    if filtertype == 'recursive':
        text = 'stepwise'
    else:
        text = 'random'

    with open(result_dir+'/Repeated_LOOCV_output_'+text+'_TM_FF.pkl', 'wb') as f:
        pickle.dump(outputs, f)
    with open(result_dir+'/Repeated_LOOCV_predoutput_'+text+'_TM_FF.pkl', 'wb') as f:
        pickle.dump(pred_outputs, f)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--filtertype', type=str, default='recursive', help='filtertype: random or recursive')
    parser.add_argument('--num_iters', type=int, default=5, help='number of iterations')
    args = parser.parse_args()
    repeated_loocv(args.filtertype,args.num_iters)