## one-to-one, one-step cross study predictions
## read README.txt for further details
## This code includes:
##   1. PCA plot for visualization of batch effect removal 
##   2. performance measurements

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score, KFold, train_test_split, GridSearchCV, LeaveOneOut
from sklearn.metrics import roc_auc_score, roc_curve, precision_recall_curve, auc
import numpy as np
from statsmodels.stats.multitest import multipletests
import scipy.stats as stat
from collections import defaultdict
import tqdm
import time

from utilities.reactome_pathway import reactome_genes
from utilities.useful_utilities import reactome_genes
from utilities.ML import ML_hyperparameters
from utilities.parse_patient_data import parse_reactomeExpression_and_immunotherapyResponse
from utilities.ML import expression_StandardScaler
from utilities.filter import across_study_filter as filter

loo = LeaveOneOut()


def across_cohort_validation(testtype):

	## Initialize
	nGene = 200
	qval = 0.01
	draw_PCA = False
	do_loocv=False
	return_coef=True
	fo_dir = './results'

	cohort_targets = {'PD1_CTLA4':['Gide'], 'PD1':['Liu','Riaz_pre','Auslander','Prat_MELANOMA'],'CTLA4':['VanAllen']}
	studies = ['Gide','Liu', 'Riaz_pre','Auslander','Prat_MELANOMA','VanAllen']
	ML_list = ['LogisticRegression']

	train_datasets = []
	for key in list(cohort_targets.keys()):
		for value in cohort_targets[key]:
			train_datasets.append(value)

	## Output dataframe
	output = defaultdict(list)
	output_col = ['train_dataset', 'test_dataset', 'ML', 'nGene', 'qval', 'test_type', 'AUC_proba', 'fpr_proba', 'tpr_proba', 'AUPRC', 'expected_AUPRC', 'precisions', 'recalls','train_auc','train_pred','train_sene_pred','train_response']

	pred_output = defaultdict(list)
	pred_output_col = ['train_dataset', 'test_dataset', 'ML', 'test_type', 'nGene', 'qval', 'sample', 'predicted_response', 'predicted_response_proba', 'observed_response','sene_pred','sene_proba']


	# Reactome pathways
	reactome = reactome_genes()

	# biomarker genes 
	bio_df = pd.read_csv('./data/Marker_summary_KEAP.txt', sep='\t')
	biomarker_dir = './data/biomarker'
	dt = 0.02
	st = .85

	## LOCO validation
	## Run cross study predictions

	#make pairs of traindatasets
	import itertools

	if testtype == 'external'
		numchoice=3
	elif testtype == 'pair':
		numchoice=2
	elif testtype == 'LOCO':
		numchoice=5

	pairs = list(itertools.combinations(train_datasets, numchoice))

	train_edfs = None
	for train_idx, train_dataset in enumerate(studies):
		### load datasets
		if ('Riaz' in train_dataset) or ('Huang' in train_dataset):
			train_samples, train_edf, train_epdf, train_responses = parse_reactomeExpression_and_immunotherapyResponse(train_dataset.split('_')[0], drug_treatment=train_dataset.split('_')[1])
		elif ('Prat' in train_dataset) & ('NSCLC' not in train_dataset):
			train_samples, train_edf, train_epdf, train_responses = parse_reactomeExpression_and_immunotherapyResponse(train_dataset.split('_')[0], Prat_cancer_type=train_dataset.split('_')[1])
		else:
			train_samples, train_edf, train_epdf, train_responses = parse_reactomeExpression_and_immunotherapyResponse(train_dataset)

		# if (len(train_samples) < 30) or (train_responses.tolist().count(1) < 10) or (train_responses.tolist().count(0) < 10):
		# 	continue
			
		if train_edfs is None: 
			train_edfs = train_edf
			train_epdfs = train_epdf
			train_responsess = train_responses
		else:
			common_genes, common_pathways = list(set(train_edf['genes'].tolist()) & set(train_edfs['genes'].tolist())), list(set(train_epdf['pathway'].tolist()) & set(train_epdfs['pathway'].tolist()))
			train_edf = train_edf.loc[train_edf['genes'].isin(common_genes),:].sort_values(by='genes').reset_index(drop=True)
			train_epdf = train_epdf.loc[train_epdf['pathway'].isin(common_pathways),:].sort_values(by='pathway').reset_index(drop=True)

			train_edfs = train_edfs.loc[train_edfs['genes'].isin(common_genes),:].sort_values(by='genes').reset_index(drop=True)
			train_epdfs = train_epdfs.loc[train_epdfs['pathway'].isin(common_pathways),:].sort_values(by='pathway').reset_index(drop=True)

			train_edfs = pd.concat([train_edfs, train_edf.iloc[:,1:]], axis=1)
			train_epdfs = pd.concat([train_epdfs, train_epdf.iloc[:,1:]], axis=1)
			train_responsess = np.concatenate([train_responsess,train_responses], axis=0)
	globalcommongenes,globalcommonpathways = common_genes, common_pathways


	for ML in ML_list:
		pathinfo = {}
		coefinfo = {}
		for pair in pairs:
			traindataset = list(pair)
			
			for test_dataset in train_datasets:
				
				if test_dataset in traindataset:
					continue

				train_edfs = None
				train_epdfs = None
				train_responsess = None

				### train dataset targets
				for key in list(cohort_targets.keys()):
					if test_dataset in cohort_targets[key]:
						test_targets = key
						break
				target = key

				batch = []
				batchnum = 0
				for train_idx, train_dataset in enumerate(traindataset):

					### load datasets
					if ('Riaz' in train_dataset) or ('Huang' in train_dataset):
						train_samples, train_edf, train_epdf, train_responses = parse_reactomeExpression_and_immunotherapyResponse(train_dataset.split('_')[0], drug_treatment=train_dataset.split('_')[1])
					elif ('Prat' in train_dataset) & ('NSCLC' not in train_dataset):
						train_samples, train_edf, train_epdf, train_responses = parse_reactomeExpression_and_immunotherapyResponse(train_dataset.split('_')[0], Prat_cancer_type=train_dataset.split('_')[1])
					else:
						train_samples, train_edf, train_epdf, train_responses = parse_reactomeExpression_and_immunotherapyResponse(train_dataset)
					if train_edfs is None: 
						train_edfs = train_edf
						train_epdfs = train_epdf
						train_responsess = train_responses
						batch += [batchnum]*(train_epdf.shape[1]-1)
						batchnum +=1
					else:
						common_genes, common_pathways = list(set(train_edf['genes'].tolist()) & set(train_edfs['genes'].tolist())), list(set(train_epdf['pathway'].tolist()) & set(train_epdfs['pathway'].tolist()))
						train_edf = train_edf.loc[train_edf['genes'].isin(common_genes),:].sort_values(by='genes').reset_index(drop=True)
						train_epdf = train_epdf.loc[train_epdf['pathway'].isin(common_pathways),:].sort_values(by='pathway').reset_index(drop=True)

						train_edfs = train_edfs.loc[train_edfs['genes'].isin(common_genes),:].sort_values(by='genes').reset_index(drop=True)
						train_epdfs = train_epdfs.loc[train_epdfs['pathway'].isin(common_pathways),:].sort_values(by='pathway').reset_index(drop=True)

						train_edfs = pd.concat([train_edfs, train_edf.iloc[:,1:]], axis=1)
						train_epdfs = pd.concat([train_epdfs, train_epdf.iloc[:,1:]], axis=1)
						train_responsess = np.concatenate([train_responsess,train_responses], axis=0)
						batch += [batchnum]*(train_epdf.shape[1]-1)
						batchnum +=1

				train_edf = train_edfs
				train_epdf = train_epdfs
				train_responses = train_responsess
				train_dataset = '_'.join(traindataset)

				if (numchoice==2)|(numchoice==3):
					if ('Riaz' in train_dataset) or ('Auslander' in train_dataset) or ('VanAllen' in train_dataset):
						## external cohorts from discovery cohorts
						continue
				
				## test set feature matching procedure

				if ('Riaz' in test_dataset) or ('Huang' in test_dataset):
					test_samples, test_edf, test_epdf, test_responses = parse_reactomeExpression_and_immunotherapyResponse(test_dataset.split('_')[0], drug_treatment=test_dataset.split('_')[1])
				elif ('Prat' in test_dataset) & ('NSCLC' not in test_dataset):
					test_samples, test_edf, test_epdf, test_responses = parse_reactomeExpression_and_immunotherapyResponse(test_dataset.split('_')[0], Prat_cancer_type=test_dataset.split('_')[1])
				else:
					test_samples, test_edf, test_epdf, test_responses = parse_reactomeExpression_and_immunotherapyResponse(test_dataset)
				
				batch += [batchnum]*(test_epdf.shape[1]-1)
				batchnum +=1
				
				print('\n\n#----------------------------------------')
				print('test data --> responder : %s / non-responder : %s'%(list(test_responses).count(1), list(test_responses).count(0)))

				### data cleanup: match genes and pathways between cohorts
				common_genes, common_pathways = list(set(train_edf['genes'].tolist()) & set(test_edf['genes'].tolist())), list(set(train_epdf['pathway'].tolist()) & set(test_epdf['pathway'].tolist()))
				train_edf = train_edf.loc[train_edf['genes'].isin(common_genes),:].sort_values(by='genes').reset_index(drop=True)
				train_epdf = train_epdf.loc[train_epdf['pathway'].isin(common_pathways),:].sort_values(by='pathway').reset_index(drop=True)
				test_edf = test_edf.loc[test_edf['genes'].isin(common_genes),:].sort_values(by='genes').reset_index(drop=True)
				test_epdf = test_epdf.loc[test_epdf['pathway'].isin(common_pathways),:].sort_values(by='pathway').reset_index(drop=True)
				
				tempconcatedf = pd.concat([train_edf.iloc[:,1:], test_edf.iloc[:,1:]], axis=1)
				tempconcatepdf = pd.concat([train_epdf.iloc[:,1:], test_epdf.iloc[:,1:]], axis=1)

				# ##### scaler
				for bat in set(batch):
					temp1 = tempconcatedf.iloc[:,np.array(batch)==bat]
					temp1 = pd.concat([train_edf['genes'], temp1], axis=1)
					tempconcatedf.iloc[:,np.array(batch)==bat] = expression_StandardScaler(temp1).iloc[:,1:]
					temp2 = tempconcatepdf.iloc[:,np.array(batch)==bat]
					temp2 = pd.concat([train_epdf['pathway'], temp2], axis=1)
					tempconcatepdf.iloc[:,np.array(batch)==bat] = expression_StandardScaler(temp2).iloc[:,1:]
				train_edf.iloc[:,1:] = tempconcatedf.iloc[:,:train_edf.shape[1]-1].values
				test_edf.iloc[:,1:] = tempconcatedf.iloc[:,train_edf.shape[1]-1:].values
				train_epdf.iloc[:,1:] = tempconcatepdf.iloc[:,:train_epdf.shape[1]-1].values
				test_epdf.iloc[:,1:] = tempconcatepdf.iloc[:,train_epdf.shape[1]-1:].values
				print(test_edf.shape)
				edf1, edf2, epdf1, epdf2 = train_edf, test_edf, train_epdf, test_epdf

				
				# if test_dataset == 'Riaz_pre':
				# 	train_epdf.to_csv('/home/mkim/Stepwise_netbio/train_epdf_Riazpre.csv',sep='\t',index=False)
				# 	test_epdf.to_csv('/home/mkim/Stepwise_netbio/test_epdf_Riazpre.csv',sep='\t',index=False)

				### network proximal pathways
				# gene expansion by network propagation results
				paths = []
				for prop_gene in [target, 'senescence']:
					bdf = pd.read_csv('%s/%s.txt'%(biomarker_dir, prop_gene), sep='\t')
					bdf = bdf.dropna(subset=['gene_id'])
					b_genes = []
					for idx, gene in enumerate(bdf.sort_values(by=['propagate_score'], ascending=False)['gene_id'].tolist()):
						if gene in train_edf['genes'].tolist():
							if not gene in b_genes:
								b_genes.append(gene)
							if len(set(b_genes)) >= nGene:
								break
					# LCC function enrichment
					tmp_hypergeom = defaultdict(list)
					pvalues, qvalues = [], []
					for pw in list(reactome.keys()):
						pw_genes = list(set(reactome[pw]) & set(train_edf['genes'].tolist()))
						M = len(train_edf['genes'].tolist())
						n = len(pw_genes)
						N = len(set(b_genes))
						k = len(set(pw_genes) & set(b_genes))
						p = stat.hypergeom.sf(k-1, M, n, N)
						tmp_hypergeom['pw'].append(pw)
						tmp_hypergeom['p'].append(p)
						pvalues.append(p)
					_, qvalues, _, _ = multipletests(pvalues)
					tmp_hypergeom['q'] = qvalues
					tmp_hypergeom = pd.DataFrame(tmp_hypergeom).sort_values(by=['q'])
					proximal_pathways = tmp_hypergeom.loc[tmp_hypergeom['q']<=qval,:]['pw'].tolist() ## proximal_pathways
					paths.append(proximal_pathways)

				proximal_pathways, senescence_pathways = paths[0], paths[1]
				target_sene_pathways = list(set(proximal_pathways) & set(senescence_pathways))
				target_sene_union_pathways = list(set(proximal_pathways) | set(senescence_pathways))

				### Train / Test dataset merging
				train_dic = {}
				test_dic = {}
				

				# 1. NetBio
				netcommon = train_epdf['pathway'].isin(proximal_pathways)
				senecommon = train_epdf['pathway'].isin(senescence_pathways)
				targetsenecommon = train_epdf['pathway'].isin(target_sene_pathways)
				targetseneunioncommon = train_epdf['pathway'].isin(target_sene_union_pathways)
				train_dic['NetBio'] = train_epdf.loc[netcommon,:]
				test_dic['NetBio'] = test_epdf.loc[netcommon,:]
				train_dic['NetBio-senescence'] = train_epdf.loc[senecommon,:]
				test_dic['NetBio-senescence'] = test_epdf.loc[senecommon,:]
				train_dic['NetBio-target_senescence'] = train_epdf.loc[targetsenecommon,:]
				test_dic['NetBio-target_senescence'] = test_epdf.loc[targetsenecommon,:]
				train_dic['NetBio+NetBio-senescence'] = train_epdf.loc[targetseneunioncommon,:]
				test_dic['NetBio+NetBio-senescence'] = test_epdf.loc[targetseneunioncommon,:]
				pathinfo[test_dataset] = train_epdf['pathway'][train_epdf['pathway'].isin(proximal_pathways)].tolist()


				# 2. controls (other biomarkers)
				for test_type, genes in zip(bio_df['Name'].tolist(), bio_df['Gene_list'].tolist()):
					genes = genes.split(':')
					train_dic[test_type] = train_edf.loc[train_edf['genes'].isin(genes),:]
					test_dic[test_type] = test_edf.loc[test_edf['genes'].isin(genes),:]
				
				
				### Measure Prediction Performances
				print('\tML training & predicting, %s'%time.ctime())
				for test_type in np.append(['NetBio','stepwise',\
								'FF-logistic','FF-RF','FF-SVC','FF-MLP',\
									'PD-L1','PD1','CTLA4','CD8T1','CAF1','T_exhaust_Pos','TAM_M2_M1_Pos','all-TME-Bio']):

					if test_type in ['NetBio']+bio_df['Name'].tolist():
						X_train, X_test = train_dic[test_type].T.values[1:], test_dic[test_type].T.values[1:]
						y_train, y_test = train_responses, test_responses
						if X_train.shape[1] * X_train.shape[0] * len(y_train) * len(y_test) == 0:
							continue
						
						# original implementation and stepwise framework comparison
						# make predictions
						model, param_grid = ML_hyperparameters(ML)
						gcv = []
						gcv = GridSearchCV(model, param_grid=param_grid, cv=5, scoring='roc_auc', n_jobs=5).fit(X_train, y_train)
						model = gcv.best_estimator_
						pred_status = gcv.best_estimator_.predict(X_test)
						pred_proba = gcv.best_estimator_.predict_proba(X_test)[:,1]
						if test_type=='NetBio':
							coef = gcv.best_estimator_.coef_[0]
							coefinfo[test_dataset] = coef
					elif test_type == 'stepwise':
						model, param_grid = ML_hyperparameters(ML)
						X_train_sene, X_test_sene = train_dic['NetBio-senescence'].T.values[1:], test_dic['NetBio-senescence'].T.values[1:]
						X_train, X_test = train_dic['NetBio'].T.values[1:], test_dic['NetBio'].T.values[1:]
						y_train, y_test = train_responses, test_responses

						if X_train.shape[1] * X_train.shape[0] * len(y_train) * len(y_test) == 0:
							continue

						if return_coef == True:
							pred_status, pred_proba, sene_pred,stepcoef,trainsene_pred = filter(train_dic,test_dic,y_train, y_test, diff_thres=dt,sens_thres=st,return_coef=return_coef)
							coefinfo[test_dataset+'_stepwise'] = stepcoef
						else: 
							pred_status, pred_proba, sene_pred,trainsene_pred = filter(train_dic,test_dic,y_train, y_test, diff_thres=dt,sens_thres=st)
						pred_status[sene_pred==1] = 0 #replace senescent samples as NR
						

						gcv = []
						gcv = GridSearchCV(model, param_grid=param_grid, cv=5, scoring='roc_auc', n_jobs=5).fit(X_train[trainsene_pred==0], y_train[trainsene_pred==0])
						X_train_sene, X_test_sene = train_dic['NetBio-senescence'].T.values[1:], test_dic['NetBio-senescence'].T.values[1:]
						y_train_sene, y_test_sene = train_responses, test_responses
						gcv_sene = GridSearchCV(model, param_grid=param_grid, cv=5, scoring='roc_auc', n_jobs=5).fit(X_train_sene, y_train_sene)
						sene_proba = gcv_sene.best_estimator_.predict_proba(X_test_sene)[:,1]
						if sene_pred==1:
							pred_proba = sene_proba

					elif test_type.startswith('FF'):
						X_train, X_test = train_dic['NetBio+NetBio-senescence'].T.values[1:], test_dic['NetBio+NetBio-senescence'].T.values[1:]
						y_train, y_test = train_responses, test_responses

						if X_train.shape[1] * X_train.shape[0] * len(y_train) * len(y_test) == 0:
							continue

						machine_learning = test_type.split('-')[1]
						ml_dict = {'logistic':'LogisticRegression', 'RF':'RandomForest', 'SVC':'SVC', 'MLP':'MLP'}
						machine_learning = ml_dict[machine_learning]

						if machine_learning != 'MLP':
							if machine_learning == 'RandomForest':
								from sklearn.ensemble import RandomForestClassifier as RFC
								model = RFC()
								param_grid = {'n_estimators':[500, 1000], 'max_depth':[X_train.shape[0]], 'class_weight':['balanced']}
							elif machine_learning == 'RandomForest2':
								from sklearn.ensemble import RandomForestClassifier as RFC
								model = RFC()
								param_grid = {'n_estimators':[1,5,10], 'max_depth':[X_train.shape[0]], 'class_weight':['balanced']}
							else:
								model, param_grid = ML_hyperparameters(machine_learning)
							gcv = GridSearchCV(model, param_grid=param_grid, scoring='roc_auc', cv=5, n_jobs=5).fit(X_train, y_train)

							pred_status = gcv.best_estimator_.predict(X_test)
							pred_proba = gcv.best_estimator_.predict_proba(X_test)[:,1]
						else:
							import warnings
							warnings.filterwarnings("ignore")

							param_grid = {}
							param_grid['learning_rate_init'] = np.arange(0.001,.01,.002)
							param_grid['alpha'] = [0.0001, 0.001, 0.01]
							param_grid['hidden_layer_sizes'] = [(X_train.shape[0]//1,),(X_train.shape[0]//2,),(X_train.shape[0]//3,),(X_train.shape[0]//2, X_train.shape[0]//3,)]
							from sklearn.neural_network import MLPClassifier as mlp
							model = mlp(max_iter=500,learning_rate='invscaling')
							gcv = GridSearchCV(model, param_grid=param_grid, scoring='roc_auc', cv=5, n_jobs=5).fit(X_train, y_train)

							pred_status = gcv.best_estimator_.predict(X_test)
							pred_proba = gcv.best_estimator_.predict_proba(X_test)[:,1]

					# pred_output
					if test_type !='stepwise':
						sene_pred = [False]*len(test_samples)
						sene_pred = np.array(sene_pred)
						zero_sene_proba = [0.0]*len(test_samples)
						zero_sene_proba = np.array(zero_sene_proba)
						for sample, pred_response, p_proba, obs_response,sene_pred_value,sene_proba_value in zip(test_samples, pred_status, pred_proba, y_test, sene_pred,zero_sene_proba):
							pred_output['train_dataset'].append(train_dataset)
							pred_output['test_dataset'].append(test_dataset)
							pred_output['ML'].append(ML)
							pred_output['test_type'].append(test_type)
							pred_output['nGene'].append(nGene)
							pred_output['qval'].append(qval)
							pred_output['sample'].append(sample)
							pred_output['predicted_response'].append(pred_response)
							pred_output['predicted_response_proba'].append(p_proba)
							pred_output['observed_response'].append(obs_response)
							pred_output['sene_pred'].append(sene_pred_value)
							pred_output['sene_proba'].append(sene_proba_value)
					else:
						for sample, pred_response, p_proba, obs_response, sene_pred_value,sene_proba_value in zip(test_samples, pred_status, pred_proba, y_test, sene_pred, sene_proba):
							pred_output['train_dataset'].append(train_dataset)
							pred_output['test_dataset'].append(test_dataset)
							pred_output['ML'].append(ML)
							pred_output['test_type'].append(test_type)
							pred_output['nGene'].append(nGene)
							pred_output['qval'].append(qval)
							pred_output['sample'].append(sample)
							pred_output['predicted_response'].append(pred_response)
							pred_output['predicted_response_proba'].append(p_proba)
							pred_output['observed_response'].append(obs_response)
							pred_output['sene_pred'].append(sene_pred_value)
							pred_output['sene_proba'].append(sene_proba_value)

					#### measure performance
					# AUC (prediction probability)
					fpr_proba, tpr_proba, _ = roc_curve(y_test, pred_proba, pos_label=1)
					AUC_proba = auc(fpr_proba, tpr_proba)
					# AUPRC
					precision, recall, _ = precision_recall_curve(y_test, pred_proba, pos_label=1)
					AUPRC = auc(recall, precision)
					expected_AUPRC = list(y_test).count(1)/len(y_test)
					output['precisions'].append(','.join(map(str, precision)))
					output['recalls'].append(','.join(map(str, recall)))

					# final results
					if (test_type == 'NetBio')| (test_type == 'stepwise'):
						print('\n\t%s, %s, train: %s, test: %s, %s'%(test_type, ML, train_dataset, test_dataset, time.ctime()))
					output['train_dataset'].append(train_dataset)
					output['test_dataset'].append(test_dataset)
					output['ML'].append(ML)
					output['nGene'].append(nGene)
					output['qval'].append(qval)
					output['test_type'].append(test_type)
					output['fpr_proba'].append(','.join(map(str, fpr_proba)))
					output['tpr_proba'].append(','.join(map(str, tpr_proba)))
					output['train_auc'].append(gcv.best_estimator_.score(X_train, y_train)) #change here 20250820
					output['train_pred'].append(gcv.best_estimator_.predict_proba(X_train)[:,1].tolist())
					output['train_sene_pred'].append(trainsene_pred.tolist() if test_type=='stepwise' else [0]*len(y_train))
					output['train_response'].append(y_train.tolist())
					for metric, score, expected_score in zip(['AUC_proba', 'AUPRC'], [AUC_proba, AUPRC], [0.5, expected_AUPRC]):
						if (test_type == 'NetBio')| (test_type == 'stepwise'):
							print('\t%s, %s -- %s : %s (random expectation=%s)'%(test_type, ML, metric, score, expected_score))
						output[metric].append(score)
						output['expected_%s'%metric].append(expected_score)
				
	output = pd.DataFrame(data=output, columns=output_col)
	output.to_csv('%s/across_study_stepwise_pair_performance_ff_fail_%s.txt'%(fo_dir, '_'.join(map(str, ML_list))),  sep='\t', index=False)
	pred_output = pd.DataFrame(data=pred_output, columns=pred_output_col)
	pred_output.to_csv('%s/across_study_stepwise_pair_prediction_ff_fail_results_perm%s_%s.txt'%(fo_dir,str(numchoice), '_'.join(map(str, ML_list))), sep='\t', index=False)
	print('Finished, %s'%time.ctime())

if __name__ == '__main__':
	import argparse
	parser = argparse.ArgumentParser()
	parser.add_argument('--testtype', type=str, default='pair', help='type of test: external, pair, LOCO')
	args = parser.parse_args()
	across_cohort_validation(testtype=args.testtype)