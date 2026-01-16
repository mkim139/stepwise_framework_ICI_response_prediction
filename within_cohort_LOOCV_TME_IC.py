
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

import tqdm
import time
from utilities.netbio_original import netbio,netbio_mlp
from utilities.filter import filter as filtering
from utilities.filter import filter_fibro as filtering_fibro

from sklearn.model_selection import cross_val_score, KFold, train_test_split, GridSearchCV, LeaveOneOut, cross_val_predict, cross_validate
from sklearn.metrics import *
from sklearn.feature_selection import VarianceThreshold, chi2, SelectKBest, f_regression
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, RidgeClassifier
loo = LeaveOneOut()

from utilities.useful_utilities import reactome_genes
from utilities.ML import ML_hyperparameters
from utilities.parse_patient_data import parse_reactomeExpression_and_immunotherapyResponse

def specificity_score(probs, threshold=0.5):
	y_pred = np.array([1 if p >= threshold else 0 for p in probs])
	return recall_score(y_true=[1]*len(y_pred), y_pred=y_pred)



result_dir = './results'
data_dir = './data/cohorts'
biomarker_dir = './data/biomarker'

patNum_cutoff = 1
num_genes = 200
qval_cutoff = 0.01
cncount = 5
num_iters = 1

finalmltype = 'LogisticRegression'
ML = 'LogisticRegression'
optimal = False

filtertype = 'recursive' #iterative enhancement
modeltesttype = 'netbio' #'netbio' ,'senescence', 'other'
filtermodel = 'senescence' #'fibro'

target_dic = {'Jung':'PD1_PD-L1', 'Liu':'PD1', 'Prat':'PD1', 
              'Gide':'PD1_CTLA4','Kim':'PD1','Mariathasan':'PD-L1',
			  'PratNSCLC':'PD1'}
datasets_to_test = target_dic.keys()
datasets_to_test = ['PratNSCLC']


# Reactome pathways
reactome = reactome_genes()

# biomarker genes
bio_df = pd.read_csv('./data/Marker_summary_KEAP.txt', sep='\t')


## LOOCV
outputs = []
study_auccompares = {}
study_rfcases = []

## output
proximity_df = defaultdict(list)
output = defaultdict(list)
output_two_step = defaultdict(list)
output_rf = defaultdict(list)
output_mlp = defaultdict(list)
output_svc = defaultdict(list)
output_logistic = defaultdict(list)

pred_output = defaultdict(list)
pred_output_two_step = defaultdict(list)
pred_output_rf = defaultdict(list)
pred_output_mlp = defaultdict(list)
pred_output_svc = defaultdict(list)
pred_output_logistic = defaultdict(list)

output_col = []
output_two_step_col = []
output_rf_col = []
output_mlp_col = []
output_svc_col = []
output_logistic_col = []

pred_output_col = []
pred_output_two_step_col = []
pred_output_mlp_col = []
pred_output_rf_col = []
pred_output_svc_col = []
pred_output_logistic_col = []

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
		
		
		sf_counts =0
		testTypes = ['PD1','PD-L1','CTLA4','CAF1','CD8T1','T_exhaust_Pos','TAM_M2_M1_Pos','all-TME-Bio','senescence','NetBio-senescence','FSS']
		for test_type in testTypes: # list(exp_dic.keys()):
			print('\n\t%s / test type : %s / ML: %s, %s'%(study, test_type, ML, time.ctime()))
			if ('Prat' in study) & (test_type=='CAF1'):
				continue

			obs_responses, pred_responses, pred_probabilities = [],[],[]
			pred_responses_two_step,pred_probabilities_two_step,obs_responses_two_step = [],[],[]
			pred_responses_rf,pred_probabilities_rf,obs_responses_rf = [],[],[]
			pred_responses_mlp,pred_probabilities_mlp,obs_responses_mlp = [],[],[]
			pred_responses_svc,pred_probabilities_svc,obs_responses_svc = [],[],[]
			pred_responses_logistic,pred_probabilities_logistic,obs_responses_logistic = [],[],[]

			testmodelaucs,filterscores = [],[]
			with tqdm.tqdm(total=exp_dic[test_type].shape[0], desc="splits", unit="split") as pbar:
				senecoefs = []
				adjcoefs = []
				ogcoefs = []
				aucs_compares = []
				for train_idx, test_idx in loo.split(exp_dic[test_type]):
					# train test split
					X_train, X_test, y_train, y_test = exp_dic[test_type][train_idx], exp_dic[test_type][test_idx], responses[train_idx], responses[test_idx]
					gcv_single = netbio(X_train,y_train,np.array([0]*len(y_train)),mltype=finalmltype)

					# predictions
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

					pbar.update(1)

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
						output_two_step[key].append(value)
					else:
						output_two_step[key].append('na')
				else:
					output_two_step[key].append(value)

			# accuracy
			print('Within-cohort performance comparisons:')
			cohort = fldr
			if cohort == 'Prat_et_al':
				cohort = 'PratMelanoma_et_al'
			accuracy = accuracy_score(obs_responses, pred_responses)
			output['accuracy'].append(accuracy)
			print('\t%s / accuracy = %s'%(cohort, accuracy))
			precision = precision_score(obs_responses, pred_responses, pos_label=1)
			output['precision'].append(precision)
			print('\t%s / precision = %s'%(cohort, precision))
			recall = recall_score(obs_responses, pred_responses, pos_label=1)
			output['recall'].append(recall)
			fpr, tpr, thresholds = roc_curve(obs_responses, pred_probabilities)
			auc_val = auc(fpr, tpr)
			output['AUC'].append(auc_val)
			print('\t%s / AUC = %s'%(cohort, auc_val))

			
			# # TP, TN, FP, FN, sensitivity, specificity
			# tn, fp, fn, tp = confusion_matrix(obs_responses, pred_responses).ravel()
			# sensitivity = tp/(tp+fn)
			# specificity = tn/(tn+fp)
			# for key, value in zip(['TP', 'TN', 'FP', 'FN', 'sensitivity', 'specificity'], [tp, tn, fp, fn, sensitivity, specificity]):
			# 	output[key].append(value)
			# 	print('\t%s / %s = %s'%(test_type, key, value))
			
			# tn, fp, fn, tp = confusion_matrix(obs_responses, pred_responses_two_step).ravel()
			# sensitivity = tp/(tp+fn)
			# specificity = tn/(tn+fp)
			# for key, value in zip(['TP', 'TN', 'FP', 'FN', 'sensitivity', 'specificity'], [tp, tn, fp, fn, sensitivity, specificity]):
			# 	output_two_step[key].append(value)
			# 	print('\t%s two step / %s = %s'%(test_type, key, value))
			
			# tn, fp, fn, tp = confusion_matrix(obs_responses_rf, pred_responses_rf).ravel()
			# sensitivity = tp/(tp+fn)
			# specificity = tn/(tn+fp)
			# for key, value in zip(['TP', 'TN', 'FP', 'FN', 'sensitivity', 'specificity'], [tp, tn, fp, fn, sensitivity, specificity]):
			# 	output_rf[key].append(value)
			# 	print('\t%s rf / %s = %s'%(test_type, key, value))
			
			# tn, fp, fn, tp = confusion_matrix(obs_responses_svc, pred_responses_svc).ravel()
			# sensitivity = tp/(tp+fn)
			# specificity = tn/(tn+fp)
			# for key, value in zip(['TP', 'TN', 'FP', 'FN', 'sensitivity', 'specificity'], [tp, tn, fp, fn, sensitivity, specificity]):
			# 	output_svc[key].append(value)
			# 	print('\t%s svc / %s = %s'%(test_type, key, value))
			
			# tn, fp, fn, tp = confusion_matrix(obs_responses_mlp, pred_responses_mlp).ravel()
			# sensitivity = tp/(tp+fn)
			# specificity = tn/(tn+fp)
			# for key, value in zip(['TP', 'TN', 'FP', 'FN', 'sensitivity', 'specificity'], [tp, tn, fp, fn, sensitivity, specificity]):
			# 	output_mlp[key].append(value)
			# 	print('\t%s mlp / %s = %s'%(test_type, key, value))
			
			# tn, fp, fn, tp = confusion_matrix(obs_responses_logistic, pred_responses_logistic).ravel()
			# sensitivity = tp/(tp+fn)
			# specificity = tn/(tn+fp)
			# for key, value in zip(['TP', 'TN', 'FP', 'FN', 'sensitivity', 'specificity'], [tp, tn, fp, fn, sensitivity, specificity]):
			# 	output_logistic[key].append(value)
			# 	print('\t%s logistic / %s = %s'%(test_type, key, value))

	output = pd.DataFrame(output)
	proximity_df = pd.DataFrame(proximity_df)


	# output = pd.DataFrame(data=output, columns=['study', 'test_type', 'ML', 'nGene', 'qval', 'accuracy', 'precision', 'recall', 'F1', 'TP', 'TN', 'FP', 'FN', 'sensitivity', 'specificity'])
	# output.to_csv('%s/LOOCV_%s_predictProba_braunlimagne1aus_nohybrid_auc2_%s_%s_%s_%s.txt'%(result_dir, ML, predict_proba,filtertype,modeltesttype,filtermodel), sep='\t', index=False)
	# output_two_step = pd.DataFrame(data=output_two_step, columns=['study', 'test_type', 'ML', 'nGene', 'qval', 'accuracy', 'precision', 'recall', 'F1', 'TP', 'TN', 'FP', 'FN', 'sensitivity', 'specificity'])
	# output_two_step.to_csv('%s/LOOCV_%s_predictProba_two_step_braunlimagne1aus_nohybrid_auc2_%s_%s_%s_%s.txt'%(result_dir, ML, predict_proba,filtertype,modeltesttype,filtermodel), sep='\t', index=False)

	pred_output = pd.DataFrame(data=pred_output, columns=['study', 'test_type', 'ML', 'nGene', 'qval', 'sample', 'predicted_response', 'pred_proba','obs_response','train_auc'])
	pred_output.to_csv(result_dir+'/' + filtermodel + '_other-pred-output.txt', sep='\t', index=False)