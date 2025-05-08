# required librairies
## pip3.12 install --force-reinstall pandas==2.2.2
## pip3.12 install --force-reinstall imbalanced-learn==0.13.0
## pip3.12 install --force-reinstall scikit-learn==1.5.2
## pip3.12 install --force-reinstall xgboost==2.1.3
## pip3.12 install --force-reinstall numpy==1.26.4
## pip3.12 install --force-reinstall joblib==1.4.2
## pip3.12 install --force-reinstall tqdm==4.67.1
## pip3.12 install --force-reinstall tqdm-joblib==0.0.4
'''
# examples of commands
## for the DT classifier
python3.12 GenomicBasedClassification.py modeling -m genomic_profiles_for_modeling.tsv -ph phenotype_dataset.tsv -o MyDirectory -x DT_FirstAnalysis -da random -s 80 -c DT -k 5 -pa tuning_parameters_DT.txt -de 20
python3.12 GenomicBasedClassification.py prediction -m genomic_profiles_for_prediction.tsv -t MyDirectory/DT_FirstAnalysis_model.obj -f MyDirectory/DT_FirstAnalysis_features.obj -ef MyDirectory/DT_FirstAnalysis_encoded_features.obj -o MyDirectory -x DT_SecondAnalysis -de 20
## for the KNN classifier
python3.12 GenomicBasedClassification.py modeling -m genomic_profiles_for_modeling.tsv -ph MyDirectory/DT_FirstAnalysis_phenotype_dataset.tsv -o MyDirectory -x KNN_FirstAnalysis -da manual -c KNN -k 5 -pa tuning_parameters_KNN.txt -de 20
python3.12 GenomicBasedClassification.py prediction -m genomic_profiles_for_prediction.tsv -t MyDirectory/KNN_FirstAnalysis_model.obj -f MyDirectory/KNN_FirstAnalysis_features.obj -ef MyDirectory/KNN_FirstAnalysis_encoded_features.obj -o MyDirectory -x KNN_SecondAnalysis -de 20
## for the LR classifier
python3.12 GenomicBasedClassification.py modeling -m genomic_profiles_for_modeling.tsv -ph MyDirectory/DT_FirstAnalysis_phenotype_dataset.tsv -o MyDirectory -x LR_FirstAnalysis -da manual -c LR -k 5 -pa tuning_parameters_LR.txt -de 20
python3.12 GenomicBasedClassification.py prediction -m genomic_profiles_for_prediction.tsv -t MyDirectory/LR_FirstAnalysis_model.obj -f MyDirectory/LR_FirstAnalysis_features.obj -ef MyDirectory/LR_FirstAnalysis_encoded_features.obj -o MyDirectory -x LR_SecondAnalysis -de 20
## for the RF classifier
python3.12 GenomicBasedClassification.py modeling -m genomic_profiles_for_modeling.tsv -ph MyDirectory/DT_FirstAnalysis_phenotype_dataset.tsv -o MyDirectory -x RF_FirstAnalysis -da manual -c RF -k 5 -pa tuning_parameters_RF.txt -de 20
python3.12 GenomicBasedClassification.py prediction -m genomic_profiles_for_prediction.tsv -t MyDirectory/RF_FirstAnalysis_model.obj -f MyDirectory/RF_FirstAnalysis_features.obj -ef MyDirectory/RF_FirstAnalysis_encoded_features.obj -o MyDirectory -x RF_SecondAnalysis -de 20
## for the SVC classifier
python3.12 GenomicBasedClassification.py modeling -m genomic_profiles_for_modeling.tsv -ph MyDirectory/DT_FirstAnalysis_phenotype_dataset.tsv -o MyDirectory -x SVC_FirstAnalysis -da manual -c SVC -k 5 -pa tuning_parameters_SVC.txt -de 20
python3.12 GenomicBasedClassification.py prediction -m genomic_profiles_for_prediction.tsv -t MyDirectory/SVC_FirstAnalysis_model.obj -f MyDirectory/SVC_FirstAnalysis_features.obj -ef MyDirectory/SVC_FirstAnalysis_encoded_features.obj -o MyDirectory -x SVC_SecondAnalysis -de 20
## for the XGB classifier
python3.12 GenomicBasedClassification.py modeling -m genomic_profiles_for_modeling.tsv -ph MyDirectory/DT_FirstAnalysis_phenotype_dataset.tsv -o MyDirectory -x XGB_FirstAnalysis -da manual -c XGB -k 2 -pa tuning_parameters_XGB.txt -de 20
python3.12 GenomicBasedClassification.py prediction -m genomic_profiles_for_prediction.tsv -t MyDirectory/XGB_FirstAnalysis_model.obj -f MyDirectory/XGB_FirstAnalysis_features.obj -ef MyDirectory/XGB_FirstAnalysis_encoded_features.obj -ec MyDirectory/XGB_FirstAnalysis_encoded_classes.obj -o MyDirectory -x XGB_SecondAnalysis -de 20
'''
# refer to guidelines to set parameters of classifiers
## decision tree classifier: DT (DecisionTreeClassifier (DT)): https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html
## k-nearest neighbors: KNN (KNeighborsClassifier()): https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html
## logistic regression: LR (LogisticRegression (LR)): https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html
## random forest: RF (RandomForestClassifier (RF)): https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html
## support vector classification: SVC (SVC()): https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html
## extreme gradient boosting: XGB (XGBClassifier (XGB)): https://xgboost.readthedocs.io/en/stable/python/python_api.html#xgboost.XGBClassifier

# import packages
## standard libraries
import sys # no individual installation because is part of the Python Standard Library (no version)
import os # no individual installation because is part of the Python Standard Library (no version)
import datetime as dt # no individual installation because is part of the Python Standard Library (no version)
import argparse as ap # no individual installation because is part of the Python Standard Library
import pickle as pi # no individual installation because is part of the Python Standard Library
import warnings as wa # no individual installation because is part of the Python Standard Library (no version)
import importlib.metadata as imp # no individual installation because is part of the Python Standard Library (no version)
## third-party libraries
import pandas as pd
import imblearn as imb
import sklearn as sk
import xgboost as xgb
import numpy as np
import joblib as jl
import tqdm as tq
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.model_selection import train_test_split, GridSearchCV, ParameterGrid
from sklearn.metrics import confusion_matrix, multilabel_confusion_matrix, classification_report, accuracy_score, precision_score, recall_score, f1_score, cohen_kappa_score, matthews_corrcoef
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from imblearn.metrics import specificity_score, sensitivity_score
from tqdm_joblib import tqdm_joblib

# set static metadata to keep outside the main function
## set workflow repositories
repositories = 'Please cite:\n GitHub (https://github.com/Nicolas-Radomski/GenomicBasedClassification),\n Docker Hub (https://hub.docker.com/r/nicolasradomski/genomicbasedclassification),\n and/or Anaconda Hub (https://anaconda.org/nicolasradomski/genomicbasedclassification).'
## set the workflow context
context = "The scikit-learn (sklearn) Python library-based workflow was inspired by an older caret R library-based version (https://doi.org/10.1186/s12864-023-09667-w), incorporating the independence of modeling (i.e. training and testing) and prediction (i.e. based on a pre-built model), the management of classification parameters, and the estimation of probabilities associated with predictions."
## set the initial workflow reference
reference = "Pierluigi Castelli, Andrea De Ruvo, Andrea Bucciacchio, Nicola D'Alterio, Cesare Camma, Adriano Di Pasquale and Nicolas Radomski (2023) Harmonization of supervised machine learning practices for efficient source attribution of Listeria monocytogenes based on genomic data. 2023, BMC Genomics, 24(560):1-19, https://doi.org/10.1186/s12864-023-09667-w"
## set the acknowledgement
acknowledgements = "Many thanks to Andrea De Ruvo and Adriano Di Pasquale for the insightful discussions that helped improve the algorithm."
# set the version and release
__version__ = "1.1.0"
__release__ = "May 2025"

# create a main function preventing the global scope from being unintentionally executed on import
def main():

	# step control
	step1_start = dt.datetime.now()

	# create the main parser
	parser = ap.ArgumentParser(
		prog="GenomicBasedClassification.py", 
		description="Perform classification-based modeling or prediction from binary (e.g. presence/absence of genes) or categorical (e.g. allele profiles) genomic data.",
		epilog=repositories
		)

	# create subparsers object
	subparsers = parser.add_subparsers(dest='subcommand')

	# create the parser for the "training" subcommand
	## get parser arguments
	parser_modeling = subparsers.add_parser('modeling', help='Help about the model building.')
	## define parser arguments
	parser_modeling.add_argument(
		'-m', '--mutations', 
		dest='inputpath_mutations', 
		action='store', 
		required=True, 
		help='Absolute or relative input path of tab-separated values (tsv) file including profiles of mutations. First column: sample identifiers identical to those in the input file of phenotypes and datasets (header: e.g. sample). Other columns: profiles of mutations (header: labels of mutations). [MANDATORY]'
		)
	parser_modeling.add_argument(
		'-ph', '--phenotypes', 
		dest='inputpath_phenotypes', 
		action='store', 
		required=True, 
		help="Absolute or relative input path of tab-separated values (tsv) file including profiles of phenotypes and datasets. First column: sample identifiers identical to those in the input file of mutations (header: e.g. sample). Second column: categorical phenotype (header: e.g. phenotype). Third column: 'training' or 'testing' dataset (header: e.g. dataset). [MANDATORY]"
		)
	parser_modeling.add_argument(
		'-da', '--dataset', 
		dest='dataset', 
		type=str,
		action='store', 
		required=False, 
		choices=['random', 'manual'], 
		default='random', 
		help="Perform random (i.e. 'random') or manual (i.e. 'manual') splitting of training and testing datasets through the holdout method. [OPTIONAL, DEFAULT: 'random']"
		)
	parser_modeling.add_argument(
		'-s', '--split', 
		dest='splitting', 
		type=int,
		action='store', 
		required=False, 
		default=None, 
		help='Percentage of random splitting to prepare the training dataset through the holdout method. [OPTIONAL, DEFAULT: None]'
		)
	parser_modeling.add_argument(
		'-l', '--limit', 
		dest='limit', 
		type=int,
		action='store', 
		required=False, 
		default=10, 
		help='Minimum number of samples per class required to estimate metrics from the training and testing datasets. [OPTIONAL, DEFAULT: 10]'
		)
	parser_modeling.add_argument(
		'-c', '--classifier', 
		dest='classifier', 
		type=str,
		action='store', 
		required=False, 
		default='XGB', 
		help='Acronym of the classifier to use among decision tree classifier (DT), k-nearest neighbors (KNN), logistic regression (LR), random forest (RF), support vector classification (SVC) or extreme gradient boosting (XGB). [OPTIONAL, DEFAULT: XGB]'
		)
	parser_modeling.add_argument(
		'-k', '--fold', 
		dest='fold', 
		type=int,
		action='store', 
		required=False, 
		default=5, 
		help='Value defining k-1 groups of samples used to train against one group of validation through the repeated k-fold cross-validation method. [OPTIONAL, DEFAULT: 5]'
		)
	parser_modeling.add_argument(
		'-pa', '--parameters', 
		dest='parameters', 
		action='store', 
		required=False, 
		help='Absolute or relative input path of a text (txt) file including tuning parameters compatible with the param_grid argument of the GridSearchCV function. (OPTIONAL)'
		)
	parser_modeling.add_argument(
		'-j', '--jobs', 
		dest='jobs', 
		type=int,
		action='store', 
		required=False, 
		default=-1, 
		help='Value defining the number of jobs to run in parallel compatible with the n_jobs argument of the GridSearchCV function. [OPTIONAL, DEFAULT: -1]'
		)
	parser_modeling.add_argument(
		'-o', '--output', 
		dest='outputpath', 
		action='store', 
		required=False, 
		default='.',
		help='Output path. [OPTIONAL, DEFAULT: .]'
		)
	parser_modeling.add_argument(
		'-x', '--prefix', 
		dest='prefix', 
		action='store', 
		required=False, 
		default='output',
		help='Prefix of output files. [OPTIONAL, DEFAULT: output]'
		)
	parser_modeling.add_argument(
		'-de', '--debug', 
		dest='debug', 
		type=int,
		action='store', 
		required=False, 
		default=0, 
		help='Traceback level when an error occurs. [OPTIONAL, DEFAULT: 0]'
		)
	parser_modeling.add_argument(
		'-w', '--warnings', 
		dest='warnings', 
		action='store_true', 
		required=False, 
		default=False, 
		help='Do not ignore warnings if you want to improve the script. [OPTIONAL, DEFAULT: False]'
		)
	parser_modeling.add_argument(
		'-nc', '--no-check', 
		dest='nocheck', 
		action='store_true', 
		required=False, 
		default=False, 
		help='Do not check versions of Python and packages. [OPTIONAL, DEFAULT: False]'
		)

	# create the parser for the "prediction" subcommand
	## get parser arguments
	parser_prediction = subparsers.add_parser('prediction', help='Help about the model-based prediction.')
	## define parser arguments
	parser_prediction.add_argument(
		'-m', '--mutations', 
		dest='inputpath_mutations', 
		action='store', 
		required=True, 
		help='Absolute or relative input path of a tab-separated values (tsv) file including profiles of mutations. First column: sample identifiers identical to those in the input file of phenotypes and datasets (header: e.g. sample). Other columns: profiles of mutations (header: labels of mutations). [MANDATORY]'
		)
	parser_prediction.add_argument(
		'-t', '--model', 
		dest='inputpath_model', 
		action='store', 
		required=True, 
		help='Absolute or relative input path of an object (obj) file including a trained scikit-learn model. [MANDATORY]'
		)
	parser_prediction.add_argument(
		'-f', '--features', 
		dest='inputpath_features', 
		action='store', 
		required=True, 
		help='Absolute or relative input path of an object (obj) file including trained scikit-learn features (i.e. mutations). [MANDATORY]'
		)
	parser_prediction.add_argument(
		'-ef', '--encodedfeatures', 
		dest='inputpath_encoded_features', 
		action='store', 
		required=True, 
		help='Absolute or relative input path of an object (obj) file including trained scikit-learn encoded features (i.e. mutations). [MANDATORY]'
		)
	parser_prediction.add_argument(
		'-ec', '--encodedclasses', 
		dest='inputpath_encoded_classes', 
		action='store', 
		required=False, 
		help='Absolute or relative input path of an object (obj) file including trained scikit-learn encoded classes (i.e. phenotypes) for the XGB model. [OPTIONAL]'
		)
	parser_prediction.add_argument(
		'-o', '--output', 
		dest='outputpath', 
		action='store', 
		required=False, 
		default='.',
		help='Absolute or relative output path. [OPTIONAL, DEFAULT: .]'
		)
	parser_prediction.add_argument(
		'-x', '--prefix', 
		dest='prefix', 
		action='store', 
		required=False, 
		default='output',
		help='Prefix of output files. [OPTIONAL, DEFAULT: output_]'
		)
	parser_prediction.add_argument(
		'-de', '--debug', 
		dest='debug', 
		type=int,
		action='store', 
		required=False, 
		default=0, 
		help='Traceback level when an error occurs. [OPTIONAL, DEFAULT: 0]'
		)
	parser_prediction.add_argument(
		'-w', '--warnings', 
		dest='warnings', 
		action='store_true', 
		required=False, 
		default=False, 
		help='Do not ignore warnings if you want to improve the script. [OPTIONAL, DEFAULT: False]'
		)
	parser_prediction.add_argument(
		'-nc', '--no-check', 
		dest='nocheck', 
		action='store_true', 
		required=False, 
		default=False, 
		help='Do not check versions of Python and packages. [OPTIONAL, DEFAULT: False]'
		)
	
	# print help if there are no arguments in the command
	if len(sys.argv)==1:
		parser.print_help()
		sys.exit(1)

	# parse the arguments
	args = parser.parse_args()

	# rename arguments
	if args.subcommand == 'modeling':
		INPUTPATH_MUTATIONS=args.inputpath_mutations
		INPUTPATH_PHENOTYPES=args.inputpath_phenotypes
		OUTPUTPATH=args.outputpath
		DATASET=args.dataset
		SPLITTING=args.splitting
		LIMIT=args.limit
		CLASSIFIER=args.classifier
		FOLD=args.fold
		PARAMETERS=args.parameters
		JOBS=args.jobs
		PREFIX=args.prefix
		DEBUG=args.debug
		WARNINGS=args.warnings
		NOCHECK=args.nocheck
	elif args.subcommand == 'prediction':
		INPUTPATH_MUTATIONS=args.inputpath_mutations
		INPUTPATH_FEATURES=args.inputpath_features
		INPUTPATH_ENCODED_FEATURES=args.inputpath_encoded_features
		INPUTPATH_ENCODED_CLASSES=args.inputpath_encoded_classes
		INPUTPATH_MODEL=args.inputpath_model
		OUTPUTPATH=args.outputpath
		PREFIX=args.prefix
		DEBUG=args.debug
		WARNINGS=args.warnings
		NOCHECK=args.nocheck

	# print a message about release
	message_release = "The GenomicBasedClassification script, version " + __version__ +  " (released in " + __release__ + ")," + " was launched"
	print(message_release)

	# set tracebacklimit
	sys.tracebacklimit = DEBUG
	message_traceback = "The traceback level was set to " + str(sys.tracebacklimit)
	print(message_traceback)

	# management of warnings
	if WARNINGS == True :
		wa.filterwarnings('default')
		message_warnings = "The warnings were not ignored"
		print(message_warnings)
	elif WARNINGS == False :
		wa.filterwarnings('ignore')
		message_warnings = "The warnings were ignored"
		print(message_warnings)

	# control versions
	if NOCHECK == False :
		## control Python version
		if sys.version_info[0] != 3 or sys.version_info[1] != 12 :
			raise Exception("Python 3.12 version is recommended")
		# control versions of packages
		if ap.__version__ != "1.1":
			raise Exception("argparse 1.1 (1.4.1) version is recommended")
		if pi.format_version != "4.0":
			raise Exception("pickle 4.0 version is recommended")
		if pd.__version__ != "2.2.2":
			raise Exception("pandas 2.2.2 version is recommended")
		if imb.__version__ != "0.13.0":
			raise Exception("imblearn 0.13.0 version is recommended")
		if sk.__version__ != "1.5.2":
			raise Exception("sklearn 1.5.2 version is recommended")
		if xgb.__version__ != "2.1.3":
			raise Exception("xgboost 2.1.3 version is recommended")
		if np.__version__ != "1.26.4":
			raise Exception("numpy 1.26.4 version is recommended")
		if jl.__version__ != "1.4.2":
			raise Exception("joblib 1.4.2 version is recommended")
		if tq.__version__ != "4.67.1":
			raise Exception("tqdm 4.67.1 version is recommended")
		if imp.version("tqdm-joblib") != "0.0.4":
			raise Exception("tqdm-joblib 0.0.4 version is recommended")
		message_versions = "The recommended versions of Python and packages were properly controlled"
	elif NOCHECK == True :
		message_versions = "The recommended versions of Python and packages were not controlled"

	# print a message about version control
	print(message_versions)

	# set rounded digits
	digits = 6

	# check the subcommand and execute corresponding code
	if args.subcommand == 'modeling':

		# print a message about subcommand
		message_subcommand = "The modeling subcommand was used"
		print(message_subcommand)

		# read input files
		## mutations
		df_mutations = pd.read_csv(INPUTPATH_MUTATIONS, sep='\t', dtype=str)
		## phenotypes
		df_phenotypes = pd.read_csv(INPUTPATH_PHENOTYPES, sep='\t', dtype=str)

		# replace missing genomic data by a string
		df_mutations = df_mutations.fillna('missing')

		# rename variables of headers
		## mutations
		df_mutations.rename(columns={df_mutations.columns[0]: 'sample'}, inplace=True)
		## phenotypes
		df_phenotypes.rename(columns={df_phenotypes.columns[0]: 'sample'}, inplace=True)
		df_phenotypes.rename(columns={df_phenotypes.columns[1]: 'phenotype'}, inplace=True)
		df_phenotypes.rename(columns={df_phenotypes.columns[2]: 'dataset'}, inplace=True)

		# sort by samples
		## mutations
		df_mutations = df_mutations.sort_values(by='sample')
		## phenotypes
		df_phenotypes = df_phenotypes.sort_values(by='sample')

		# indentify the type of phenotype classes
		## count the phenotype classes
		### count each phenotype classes
		counts_each_classes_series = df_phenotypes.groupby('phenotype').size()
		### count classes
		counts_classes_int = len(counts_each_classes_series.index)
		### retrieve phenotype classes as string
		classes_str = str(counts_each_classes_series.index.astype(str).tolist()).replace("[", "").replace("]", "")
		### define the type of phenotype classes
		if counts_classes_int == 2:
			type_phenotype_classes = 'two classes'
			message_number_phenotype_classes = "The provided phenotype harbored " + str(counts_classes_int) + " classes: " + classes_str
			print(message_number_phenotype_classes)
		elif counts_classes_int > 2:
			type_phenotype_classes = 'more than two classes'
			message_number_phenotype_classes = "The provided phenotype harbored " + str(counts_classes_int) + " classes: " + classes_str
			print(message_number_phenotype_classes)
		elif counts_classes_int == 1:
			message_number_phenotype_classes = "The provided phenotype classes must be higher or equal to two"
			raise Exception(message_number_phenotype_classes)

		# define minimal limites of samples (i.e. 2 * counts_classes_int * LIMIT per class)
		limit_samples = 2 * counts_classes_int * LIMIT

		# check the input file of mutations
		## calculate the number of rows
		rows_mutations = len(df_mutations)
		## calculate the number of columns
		columns_mutations = len(df_mutations.columns)
		## check if more than limit_samples rows and 4 columns
		if (rows_mutations >= limit_samples) and (columns_mutations >= 4): 
			message_input_mutations = "The number of recommended rows (i.e. >= " + str(limit_samples) + ") and expected columns (i.e. >= 4) of the input file of mutations was properly controled (i.e. " + str(rows_mutations) + " and " + str(columns_mutations) + " , respectively)"
			print (message_input_mutations)
		else: 
			message_input_mutations = "The number of recommended rows (i.e. >= " + str(limit_samples) + ") and expected columns (i.e. >= 4) of the input file of mutations was inproperly controled (i.e. " + str(rows_mutations) + " and " + str(columns_mutations) + " , respectively)"
			raise Exception(message_input_mutations)

		# check the input file of phenotypes
		## calculate the number of rows
		rows_phenotypes = len(df_phenotypes)
		## calculate the number of columns
		columns_phenotypes = len(df_phenotypes.columns)
		## check if more than limit_samples rows and 4 columns
		if (rows_phenotypes >= limit_samples) and (columns_phenotypes == 3): 
			message_input_phenotypes = "The number of recommended rows (i.e. >= " + str(limit_samples) + ") and expected columns (i.e. = 3) of the input file of phenotypes was properly controled (i.e. " + str(rows_phenotypes) + " and " + str(columns_phenotypes) + " , respectively)"
			print (message_input_phenotypes)
		else: 
			message_input_phenotypes = "The number of recommended rows (i.e. >= " + str(limit_samples) + ") and expected columns (i.e. = 3) of the input file of phenotypes was inproperly controled (i.e. " + str(rows_phenotypes) + " and " + str(columns_phenotypes) + " , respectively)"
			raise Exception(message_input_phenotypes)
		## check the absence of missing data in the second column (i.e. phenotype)
		missing_phenotypes = pd.Series(df_phenotypes.iloc[:,1]).isnull().values.any()
		if missing_phenotypes == False: 
			message_missing_phenotypes = "The absence of missing phenotypes in the input file of phenotypes was properly controled (i.e. the second column)"
			print (message_missing_phenotypes)
		elif missing_phenotypes == True:
			message_missing_phenotypes = "The absence of missing phenotypes in the input file of phenotypes was inproperly controled (i.e. the second column)"
			raise Exception(message_missing_phenotypes)
		## check the absence of values other than 'training' or 'testing' in the third column (i.e. dataset)
		if (DATASET == "manual"):
			expected_datasets = all(df_phenotypes.iloc[:,2].isin(["training", "testing"]))
			if expected_datasets == True: 
				message_expected_datasets = "The expected datasets (i.e. 'training' or 'testing') in the input file of phenotypes were properly controled (i.e. the third column)"
				print (message_expected_datasets)
			elif expected_datasets == False:
				message_expected_datasets = "The expected datasets (i.e. 'training' or 'testing') in the input file of phenotypes were inproperly controled (i.e. the third column)"
				raise Exception(message_expected_datasets)
		elif (DATASET == "random"):
			message_expected_datasets = "The expected datasets (i.e. 'training' or 'testing') in the input file of phenotypes were not controled (i.e. the third column)"
			print (message_expected_datasets)

		# check if lists of sorted samples are identical
		## convert DataFrame column as a list
		lst_mutations = df_mutations['sample'].tolist()
		lst_phenotypes = df_phenotypes['sample'].tolist()
		## compare lists
		if lst_mutations == lst_phenotypes: 
			message_sample_identifiers = "The sorted sample identifiers were confirmed as identical between the input files of mutations and phenotypes/datasets"
			print (message_sample_identifiers)
		else: 
			message_sample_identifiers = "The sorted sample identifiers were confirmed as not identical between the input files of mutations and phenotypes/datasets"
			raise Exception(message_sample_identifiers)

		# transform the phenotype classes into phenotype numbers for the XGB model
		if CLASSIFIER == 'XGB':
			le = LabelEncoder()
			df_phenotypes["phenotype"] = le.fit_transform(df_phenotypes["phenotype"])
			encoded_classes = le.classes_
			message_class_encoder = "The phenotype classes were encoded for the XGB classifier (i.e. 0, 1, 2 ....): " + str(encoded_classes)
			print(message_class_encoder)
		else:
			message_class_encoder = "The phenotype classes were not encoded for the classifiers other than the XGB classifier"
			print(message_class_encoder)

		# check compatibility between the dataset and splitting arguments
		if (DATASET == 'random') and (SPLITTING != None):
			message_compatibility_dataset_slitting = "The provided selection of training/testing datasets (i.e. " + DATASET + ") and percentage of random splitting (i.e. " + str(SPLITTING) + "%) were compatible"
			print(message_compatibility_dataset_slitting)
		elif (DATASET == 'random') and (SPLITTING == None):
			message_compatibility_dataset_slitting = "The provided selection of training/testing datasets (i.e. " + DATASET + ") required the percentage of random splitting (i.e. " + str(SPLITTING) + ")"
			raise Exception(message_compatibility_dataset_slitting)
		elif (DATASET == 'manual') and (SPLITTING == None):
			message_compatibility_dataset_slitting = "The provided selection of training/testing datasets (i.e. " + DATASET + ") and percentage of random splitting (i.e. " + str(SPLITTING) + ") were compatible"
			print(message_compatibility_dataset_slitting)
		elif (DATASET == 'manual') and (SPLITTING != None):
			message_compatibility_dataset_slitting = "The provided selection of training/testing datasets (i.e. " + DATASET + ") did not require the percentage of random splitting (i.e. " + str(SPLITTING) + "%)"
			raise Exception(message_compatibility_dataset_slitting)

		# perform splitting of the traning and testing datasets according to the setting
		if DATASET == 'random':
			message_dataset = "The training and testing datasets were constructed based on the 'random' setting"
			print(message_dataset)
			# trash dataset column
			df_phenotypes = df_phenotypes.drop("dataset", axis='columns')
			# index with sample identifiers the dataframes mutations (X) and phenotypes (y)
			X = df_mutations.set_index('sample')
			y = df_phenotypes.set_index('sample')
			# split the dataset into training and testing sets
			X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, train_size = SPLITTING/100) # random_state=42 for reproducible
		elif DATASET == 'manual':
			message_dataset = "The training and testing datasets were constructed based on the 'manual' setting"
			print(message_dataset)
			# concatenate horizontally phenotypes and mutations dataframes
			df_phenotypes_mutations = pd.concat([df_phenotypes, df_mutations], axis=1, ignore_index=False)
			# extract training and testing dataframes
			df_training, df_testing = [x for _, x in df_phenotypes_mutations.groupby(df_phenotypes_mutations['dataset'] == 'testing')]
			# retrieve the phenotype dataframe
			df_phenotypes_training = df_training.iloc[:,0:2]
			df_phenotypes_testing = df_testing.iloc[:,0:2]
			# retrieve the mutation dataframe
			df_mutations_training = df_training.iloc[:,3:]
			df_mutations_testing = df_testing.iloc[:,3:]
			# index with sample identifiers the dataframes mutations (X) and phenotypes (y)
			X_train = df_mutations_training.set_index('sample')
			X_test = df_mutations_testing.set_index('sample')
			y_train = df_phenotypes_training.set_index('sample')
			y_test = df_phenotypes_testing.set_index('sample')

		# check number of samples per class
		## retrieve a list of unique classes
		### transform a dataframe column into a list
		classes_unique_lst = df_phenotypes['phenotype'].tolist()
		### remove replicates
		classes_unique_lst = list(set(classes_unique_lst))
		### sort by alphabetic order
		classes_unique_lst.sort()
		## retieve a list of classes in the whole dataset
		classes_dataset_lst = df_phenotypes['phenotype'].tolist()
		## retieve a list of classes in the training dataset
		classes_train_lst = y_train['phenotype'].tolist()
		## retieve a list of classes in the testing dataset
		classes_test_lst = y_test['phenotype'].tolist()
		## count classes
		### in the whole dataset 
		count_dataset_lst = []
		for element in classes_unique_lst: 
			count = 0
			count += classes_dataset_lst.count(element)
			count_dataset_lst.append(count)
		### in the training dataset
		count_train_lst = []
		for element in classes_unique_lst: 
			count = 0
			count += classes_train_lst.count(element)
			count_train_lst.append(count)
		### in the testing dataset
		count_test_lst = []
		for element in classes_unique_lst: 
			count = 0
			count += classes_test_lst.count(element)
			count_test_lst.append(count)
		## combine horizontally lists into a dataframe
		count_classes_df = pd.DataFrame({
			'phenotype': classes_unique_lst,
			'dataset': count_dataset_lst,
			'training': count_train_lst,
			'testing': count_test_lst
			})
		## control minimal number of samples per class
		### detect small number of samples in the training dataset
		detection_train = all(element >= LIMIT for element in count_classes_df['training'].tolist())
		### detect small number of samples in the testing dataset
		detection_test = all(element >= LIMIT for element in count_classes_df['testing'].tolist())
		### check the minimal quantity of samples per class
		if (detection_train == True) and (detection_test == True):
			message_count_classes = "The number of samples per class in the training and testing datasets was properly controlled to be higher than the set limit (i.e. " + str(LIMIT) + ")"
			print(message_count_classes)
		elif (detection_train == False) or (detection_test == False):
			message_count_classes = "The number of samples per class in the training and testing datasets was improperly controlled, making it lower than the set limit (i.e. " + str(LIMIT) + ")"
			print(count_classes_df)
			raise Exception(message_count_classes)

		# encode string features into binary features for the training dataset
		## instantiate encoder object
		encoder = OneHotEncoder(handle_unknown = 'ignore', sparse_output = False).set_output(transform='pandas')
		## transform data with the encoder into an array for the training dataset
		X_train_encoded = encoder.fit_transform(X_train[X_train.columns])
		## save the encoded features
		encoded_features = encoder.categories_

		# encode string features into binary features for the testing dataset
		## instantiate encoder object
		encoder = OneHotEncoder(handle_unknown = 'ignore', sparse_output = False, categories = encoded_features).set_output(transform='pandas')
		## retrieve features
		features = X_train.columns
		## transform data with the encoder into an array for the testing dataset using produced encoded features
		X_test_encoded = encoder.fit_transform(X_test[features])

		# prepare the model
		## initializing the model
		if CLASSIFIER == 'DT':
			message_classifier = "The provided classifier was properly recognized: decision tree classifier (DT)"
			print(message_classifier)
			selected_classifier = DecisionTreeClassifier() # decision tree classifier (DT)
		elif CLASSIFIER == 'KNN':
			message_classifier = "The provided classifier was properly recognized: k-nearest neighbors (KNN)"
			print(message_classifier)
			selected_classifier = KNeighborsClassifier() # k-nearest neighbors (KNN)
		elif CLASSIFIER == 'LR':
			message_classifier = "The provided classifier was properly recognized: logistic regression (LR)"
			print(message_classifier)
			selected_classifier = LogisticRegression() # logistic regression (LR)
		elif CLASSIFIER == 'RF':
			message_classifier = "The provided classifier was properly recognized: random forest (RF)"
			print(message_classifier)
			selected_classifier = RandomForestClassifier() # random forest (RF)
		elif CLASSIFIER == 'SVC':
			message_classifier = "The provided classifier was properly recognized: support vector classification (SVC)"
			print(message_classifier)
			selected_classifier = SVC(probability = True) # support vector classification (SVC)
		elif CLASSIFIER == 'XGB':
			message_classifier = "The provided classifier was properly recognized: extreme gradient boosting (XGB)"
			print(message_classifier)
			if type_phenotype_classes == 'more than two classes':
				selected_classifier = xgb.XGBClassifier(objective = 'multi:softmax') # extreme gradient boosting (XGB)
				message_XGB_type_phenotype_classes = "The XGB classifier was set to manage " + type_phenotype_classes + " phenotype classes"
				print(message_XGB_type_phenotype_classes)
			elif type_phenotype_classes == 'two classes':
				selected_classifier = xgb.XGBClassifier(objective = 'binary:logistic') # extreme gradient boosting (XGB)
				message_XGB_type_phenotype_classes = "The XGB classifier was set to manage " + type_phenotype_classes + " phenotype classes"
				print(message_XGB_type_phenotype_classes)
		else: 
			message_classifier = "The provided classifier is not implemented yet"
			raise Exception(message_classifier)
		## initializing the classifier parameters if the tuning parameters are not provided by the user
		if PARAMETERS == None:
			parameters = [{}]
			message_parameters = "The tuning parameters were not provided by the user"
			print(message_parameters)
		## initializing the classifier parameters if the tuning parameters are provided by the user
		elif PARAMETERS != None:
			### open the provided file of tuning parameters
			parameters_file = open(PARAMETERS, "r")
			### read provided tuning parameters and convert string dictionary to dictionary
			parameters = [eval(parameters_file.read())]
			### close the provided file of tuning parameters
			parameters_file.close()
			### print message
			message_parameters = "The tuning parameters were provided by the user: " + str(parameters).replace("[", "").replace("]", "")
			print(message_parameters)
		
		# build the model
		## prepare the grid search cross-validation (CV) first
		model = GridSearchCV(
			estimator=selected_classifier,
			param_grid=parameters,
			cv=FOLD,
			scoring='accuracy',
			verbose=0,
			n_jobs=JOBS
		)
		## compute the total number of fits in GridSearchCV
		param_combinations = len(list(ParameterGrid(parameters))) * FOLD
		## fit the model
		### use a tqdm progress bar from the tqdm_joblib library (compatible with GridSearchCV)
		### use a tqdm progress bar immediately after the last print (position=0), disable the additional bar after completion (leave=False), and allow for dynamic resizing (dynamic_ncols=True)
		### force GridSearchCV to use the threading backend to avoid the DeprecationWarning from fork and ChildProcessError from the loky backend (default in joblib)
		### threading is slower than loky, but it allows using a progress bar with GridSearchCV and avoids the DeprecationWarning and ChildProcessError
		with tq.tqdm(total=param_combinations, desc="Model building progress", position=0, leave=False, dynamic_ncols=True) as progress_bar:
			with jl.parallel_backend('threading', n_jobs=JOBS):
				with tqdm_joblib(progress_bar):
					model.fit(X_train_encoded, y_train["phenotype"].values.ravel())
		## print best parameters
		if PARAMETERS == None:
			message_best_parameters = "The best parameters during model cross-validation were not computed because they were not provided"
		elif PARAMETERS != None:
			message_best_parameters = "The best parameters during model cross-validation were: " + str(model.best_params_)
		print(message_best_parameters)
		## print best score
		message_best_score = "The best accuracy during model cross-validation was: " + str(round(model.best_score_, digits))
		print(message_best_score)
		
		# retrieve the combinations of tested parameters and corresponding scores
		## combinations of tested parameters
		allparameters_lst = model.cv_results_['params']
		## corresponding scores
		allscores_nda = model.cv_results_['mean_test_score']
		## transform the list of parameters into a dataframe
		allparameters_df = pd.DataFrame({'parameters': allparameters_lst})
		## transform the ndarray of scores into a dataframe
		allscores_df = pd.DataFrame({'scores': allscores_nda})
		## concatenate horizontally dataframes
		all_scores_parameters_df = pd.concat([allscores_df, allparameters_df], axis=1, ignore_index=False)
		## remove unnecessary characters
		### replace each dictionary by string
		all_scores_parameters_df['parameters'] = all_scores_parameters_df['parameters'].apply(lambda x: str(x))
		### replace special characters { and } by nothing
		all_scores_parameters_df['parameters'] = all_scores_parameters_df['parameters'].replace(r'[\{\}]', '', regex=True)

		# select the best model
		best_model = model.best_estimator_

		## from the training dataset
		y_pred_train = best_model.predict(X_train_encoded)
		## from the testing dataset
		y_pred_test = best_model.predict(X_test_encoded)

		# retrieve trained phenotype classes
		if CLASSIFIER == 'XGB':
			classes_lst = encoded_classes
		else:
			classes_lst = best_model.classes_

		# extract the confusion matrices (cm)
		## retrieve classes
		### transform numpy.ndarray into pandas.core.frame.DataFrame
		classes_df = pd.DataFrame(classes_lst)
		### rename variables of headers
		classes_df.rename(columns={0: 'phenotype'}, inplace=True)
		## extract confusion matrix from the training dataset
		### get the confusion matrix
		cm_classes_train_nda = confusion_matrix(y_train, y_pred_train)
		### transform numpy.ndarray into pandas.core.frame.DataFrame
		cm_classes_train_df = pd.DataFrame(cm_classes_train_nda, columns = classes_lst)
		### concatenate horizontally classes and confusion matrix
		cm_classes_train_df = pd.concat([classes_df, cm_classes_train_df], axis=1)
		## extract confusion matrix from the testing dataset
		### get the confusion matrix
		cm_classes_test_nda = confusion_matrix(y_test, y_pred_test)
		### transform numpy.ndarray into pandas.core.frame.DataFrame
		cm_classes_test_df = pd.DataFrame(cm_classes_test_nda, columns = classes_lst)
		### concatenate horizontally classes and confusion matrix
		cm_classes_test_df = pd.concat([classes_df, cm_classes_test_df], axis=1)

		# extract true positive (TP), true negative (TN), false positive (FP) and false negative (FN) for each class
		# |      |PRED |
		# |      |- |+ |
		# |EXP |-|TN|FP|
		# |EXP |+|FN|TP|
		## from the training dataset
		### compute confusion matrix (cm)
		cm_metrics_train_nda = multilabel_confusion_matrix(y_train, y_pred_train)
		### create an empty list
		metrics_train_lst = []
		### loop over numpy.ndarray and classes
		for nda, classes in zip(cm_metrics_train_nda, classes_lst):
			##### extract TN, FP, FN and TP			
			tn_train, fp_train, fn_train, tp_train = nda.ravel()
			##### create dataframes
			metrics_classes_train_df = pd.DataFrame({'phenotype': classes, 'TN': [int(tn_train)], 'FP': [int(fp_train)], 'FN': [int(fn_train)], 'TP': [int(tp_train)]})
			##### add dataframes into a list
			metrics_train_lst.append(metrics_classes_train_df)
		### concatenate vertically dataframes
		metrics_classes_train_df = pd.concat(metrics_train_lst, axis=0, ignore_index=True)
		## from the testing dataset
		### compute confusion matrix (cm)
		cm_metrics_test_nda = multilabel_confusion_matrix(y_test, y_pred_test)
		### create an empty list
		metrics_test_lst = []
		### loop over numpy.ndarray and classes
		for nda, classes in zip(cm_metrics_test_nda, classes_lst):
			##### extract TN, FP, FN and TP			
			tn_test, fp_test, fn_test, tp_test = nda.ravel()
			##### create dataframes
			metrics_classes_test_df = pd.DataFrame({'phenotype': classes, 'TN': [int(tn_test)], 'FP': [int(fp_test)], 'FN': [int(fn_test)], 'TP': [int(tp_test)]})
			##### add dataframes into a list
			metrics_test_lst.append(metrics_classes_test_df)
		### concatenate vertically dataframes
		metrics_classes_test_df = pd.concat(metrics_test_lst, axis=0, ignore_index=True)

		# calculate the class-dependent metrics safely refactoring metrics calculations using np.where to avoid division by zero (i.e. result is safely set to 0 instead of nan or inf)
		# support = TP+FN
		# accuracy = (TP+TN)/(TP+FP+FN+TN)
		# sensitivity = TP/(TP+FN)
		# specificity = TN/(TN+FP)
		# precision = TP/(TP+FP)
		# recall = TP/(TP+FN)
		# f1-score = 2*(precision*recall)/(precision+recall)
		# MCC (Matthews Correlation Coefficient) = (TP*TN-FP*FN)/sqrt((TP+FP)*(TP+FN)*(TN+FP)*(TN+FN))
		# Cohen's kappa is not well-defined per class, as it is designed for overall pairwise agreement across the confusion matrix
		## from the training dataset
		metrics_classes_train_df['support'] = (metrics_classes_train_df.TP + metrics_classes_train_df.FN
		)
		train_acc_denom = metrics_classes_train_df.TP + metrics_classes_train_df.FP + metrics_classes_train_df.FN + metrics_classes_train_df.TN
		metrics_classes_train_df['accuracy'] = np.where(
			train_acc_denom == 0, 0,
			round((metrics_classes_train_df.TP + metrics_classes_train_df.TN) / train_acc_denom, digits)
		)
		train_sens_denom = metrics_classes_train_df.TP + metrics_classes_train_df.FN
		metrics_classes_train_df['sensitivity'] = np.where(
			train_sens_denom == 0, 0,
			round(metrics_classes_train_df.TP / train_sens_denom, digits)
		)
		train_spec_denom = metrics_classes_train_df.TN + metrics_classes_train_df.FP
		metrics_classes_train_df['specificity'] = np.where(
			train_spec_denom == 0, 0,
			round(metrics_classes_train_df.TN / train_spec_denom, digits)
		)
		train_prec_denom = metrics_classes_train_df.TP + metrics_classes_train_df.FP
		metrics_classes_train_df['precision'] = np.where(
			train_prec_denom == 0, 0,
			round(metrics_classes_train_df.TP / train_prec_denom, digits)
		)
		train_recall_denom = metrics_classes_train_df.TP + metrics_classes_train_df.FN
		metrics_classes_train_df['recall'] = np.where(
			train_recall_denom == 0, 0,
			round(metrics_classes_train_df.TP / train_recall_denom, digits)
		)
		train_f1_denom = metrics_classes_train_df.precision + metrics_classes_train_df.recall
		metrics_classes_train_df['f1-score'] = np.where(
			train_f1_denom == 0, 0,
			round(2 * metrics_classes_train_df.precision * metrics_classes_train_df.recall / train_f1_denom, digits)
		)
		train_mcc_denom = np.sqrt(
			(metrics_classes_train_df.TP + metrics_classes_train_df.FP) *
			(metrics_classes_train_df.TP + metrics_classes_train_df.FN) *
			(metrics_classes_train_df.TN + metrics_classes_train_df.FP) *
			(metrics_classes_train_df.TN + metrics_classes_train_df.FN)
		)
		metrics_classes_train_df['MCC'] = np.where(
			train_mcc_denom == 0, 0,
			round((metrics_classes_train_df.TP * metrics_classes_train_df.TN - metrics_classes_train_df.FP * metrics_classes_train_df.FN) / train_mcc_denom, digits)
		)
		## from the testing dataset
		metrics_classes_test_df['support'] = (metrics_classes_test_df.TP + metrics_classes_test_df.FN
		)
		test_acc_denom = metrics_classes_test_df.TP + metrics_classes_test_df.FP + metrics_classes_test_df.FN + metrics_classes_test_df.TN
		metrics_classes_test_df['accuracy'] = np.where(
			test_acc_denom == 0, 0,
			round((metrics_classes_test_df.TP + metrics_classes_test_df.TN) / test_acc_denom, digits)
		)
		test_sens_denom = metrics_classes_test_df.TP + metrics_classes_test_df.FN
		metrics_classes_test_df['sensitivity'] = np.where(
			test_sens_denom == 0, 0,
			round(metrics_classes_test_df.TP / test_sens_denom, digits)
		)
		test_spec_denom = metrics_classes_test_df.TN + metrics_classes_test_df.FP
		metrics_classes_test_df['specificity'] = np.where(
			test_spec_denom == 0, 0,
			round(metrics_classes_test_df.TN / test_spec_denom, digits)
		)
		test_prec_denom = metrics_classes_test_df.TP + metrics_classes_test_df.FP
		metrics_classes_test_df['precision'] = np.where(
			test_prec_denom == 0, 0,
			round(metrics_classes_test_df.TP / test_prec_denom, digits)
		)
		test_recall_denom = metrics_classes_test_df.TP + metrics_classes_test_df.FN
		metrics_classes_test_df['recall'] = np.where(
			test_recall_denom == 0, 0,
			round(metrics_classes_test_df.TP / test_recall_denom, digits)
		)
		test_f1_denom = metrics_classes_test_df.precision + metrics_classes_test_df.recall
		metrics_classes_test_df['f1-score'] = np.where(
			test_f1_denom == 0, 0,
			round(2 * metrics_classes_test_df.precision * metrics_classes_test_df.recall / test_f1_denom, digits)
		)
		test_mcc_denom = np.sqrt(
			(metrics_classes_test_df.TP + metrics_classes_test_df.FP) *
			(metrics_classes_test_df.TP + metrics_classes_test_df.FN) *
			(metrics_classes_test_df.TN + metrics_classes_test_df.FP) *
			(metrics_classes_test_df.TN + metrics_classes_test_df.FN)
		)
		metrics_classes_test_df['MCC'] = np.where(
			test_mcc_denom == 0, 0,
			round((metrics_classes_test_df.TP * metrics_classes_test_df.TN - metrics_classes_test_df.FP * metrics_classes_test_df.FN) / test_mcc_denom, digits)
		)

		# calculate the global metrics safely refactoring metrics calculations using np.nan_to_num to avoid division by zero (i.e. result is safely set to 0 instead of nan or inf)
		## from the training dataset
		accuracy_train = round(np.nan_to_num(accuracy_score(y_train, y_pred_train)), digits)
		sensitivity_train = round(np.nan_to_num(sensitivity_score(y_train, y_pred_train, average='macro')), digits)
		specificity_train = round(np.nan_to_num(specificity_score(y_train, y_pred_train, average='macro')), digits)
		precision_train = round(np.nan_to_num(precision_score(y_train, y_pred_train, average='macro', zero_division=0)), digits)
		recall_train = round(np.nan_to_num(recall_score(y_train, y_pred_train, average='macro', zero_division=0)), digits)
		f1_score_train = round(np.nan_to_num(f1_score(y_train, y_pred_train, average='macro', zero_division=0)), digits)
		mcc_train = round(np.nan_to_num(matthews_corrcoef(y_train, y_pred_train)), digits)
		cohen_kappa_train = round(np.nan_to_num(cohen_kappa_score(y_train, y_pred_train)), digits)
		## from the testing dataset
		accuracy_test = round(np.nan_to_num(accuracy_score(y_test, y_pred_test)), digits)
		sensitivity_test = round(np.nan_to_num(sensitivity_score(y_test, y_pred_test, average='macro')), digits)
		specificity_test = round(np.nan_to_num(specificity_score(y_test, y_pred_test, average='macro')), digits)
		precision_test = round(np.nan_to_num(precision_score(y_test, y_pred_test, average='macro', zero_division=0)), digits)
		recall_test = round(np.nan_to_num(recall_score(y_test, y_pred_test, average='macro', zero_division=0)), digits)
		f1_score_test = round(np.nan_to_num(f1_score(y_test, y_pred_test, average='macro', zero_division=0)), digits)
		mcc_test = round(np.nan_to_num(matthews_corrcoef(y_test, y_pred_test)), digits)
		cohen_kappa_test = round(np.nan_to_num(cohen_kappa_score(y_test, y_pred_test)), digits)
		## combine in dataframes
		metrics_global_train_df = pd.DataFrame({
			'accuracy': [round(accuracy_train, digits)], 
			'sensitivity': [round(sensitivity_train, digits)], 
			'specificity': [round(specificity_train, digits)], 
			'precision': [round(precision_train, digits)], 
			'recall': [round(recall_train, digits)], 
			'f1-score': [round(f1_score_train, digits)], 
			'MCC': [round(mcc_train, digits)], 
			'Cohen kappa': [round(cohen_kappa_train, digits)]
			})
		metrics_global_test_df = pd.DataFrame({
			'accuracy': [round(accuracy_test, digits)], 
			'sensitivity': [round(sensitivity_test, digits)], 
			'specificity': [round(specificity_test, digits)], 
			'precision': [round(precision_test, digits)], 
			'recall': [round(recall_test, digits)], 
			'f1-score': [round(f1_score_test, digits)], 
			'MCC': [round(mcc_test, digits)], 
			'Cohen kappa': [round(cohen_kappa_test, digits)]
			})

		# combine expectations and predictions from the training dataset
		## transform numpy.ndarray into pandas.core.frame.DataFrame
		y_pred_train_df = pd.DataFrame(y_pred_train)
		## retrieve the sample index in a column
		y_train_df = y_train.reset_index().rename(columns={"index":"sample"})
		## concatenate horizontally with reset index
		combined_train_df = pd.concat([y_train_df.reset_index(drop=True), y_pred_train_df.reset_index(drop=True)], axis=1)
		## rename variables of headers
		combined_train_df.rename(columns={'phenotype': 'expectation'}, inplace=True)
		combined_train_df.rename(columns={0: 'prediction'}, inplace=True)

		# combine expectations and predictions from the testing dataset
		## transform numpy.ndarray into pandas.core.frame.DataFrame
		y_pred_test_df = pd.DataFrame(y_pred_test)
		## retrieve the sample index in a column
		y_test_df = y_test.reset_index().rename(columns={"index":"sample"})
		## concatenate horizontally with reset index
		combined_test_df = pd.concat([y_test_df.reset_index(drop=True), y_pred_test_df.reset_index(drop=True)], axis=1)
		## rename variables of headers
		combined_test_df.rename(columns={'phenotype': 'expectation'}, inplace=True)
		combined_test_df.rename(columns={0: 'prediction'}, inplace=True)
		## transform back the phenotype numbers into phenotype classes for the XGB model
		if CLASSIFIER == 'XGB':
			combined_train_df["expectation"] = le.inverse_transform(combined_train_df["expectation"])
			combined_train_df["prediction"] = le.inverse_transform(combined_train_df["prediction"])
			combined_test_df["expectation"] = le.inverse_transform(combined_test_df["expectation"])
			combined_test_df["prediction"] = le.inverse_transform(combined_test_df["prediction"])
		
		# retrieve p-values
		## as a numpy.ndarray from the training dataset
		y_pvalues_train_nda = best_model.predict_proba(X_train_encoded)
		## as a numpy.ndarray from the testing dataset
		y_pvalues_test_nda = best_model.predict_proba(X_test_encoded)
		## transform numpy.ndarray into pandas.core.frame.DataFrame
		y_pvalues_train_df = pd.DataFrame(y_pvalues_train_nda, columns = classes_lst)
		y_pvalues_test_df = pd.DataFrame(y_pvalues_test_nda, columns = classes_lst)

		# concatenate horizontally the combined predictions and p-values without reset index
		## from the training dataset
		combined_train_df = pd.concat([combined_train_df, y_pvalues_train_df], axis=1)
		## from the testing dataset
		combined_test_df = pd.concat([combined_test_df, y_pvalues_test_df], axis=1)

		# round digits of a dataframe avoid SettingWithCopyWarning with .copy()
		## from the training dataset
		combined_train_df = combined_train_df.copy()
		combined_train_df[combined_train_df.select_dtypes(include='number').columns] = combined_train_df.select_dtypes(include='number').round(digits)
		## from the testing dataset
		combined_test_df = combined_test_df.copy()
		combined_test_df[combined_test_df.select_dtypes(include='number').columns] = combined_test_df.select_dtypes(include='number').round(digits)

		# combine phenotypes and datasets to potentially use it as future input
		## select columns of interest with .copy() to prevents potential SettingWithCopyWarning
		simplified_train_df = combined_train_df.iloc[:,0:2].copy()
		simplified_test_df = combined_test_df.iloc[:,0:2].copy()
		## add a column
		simplified_train_df['dataset'] = 'training'
		simplified_test_df['dataset'] = 'testing'
		## concatenate vertically dataframes
		simplified_train_test_df = pd.concat([simplified_train_df, simplified_test_df], axis=0, ignore_index=True)
		## rename variables of header
		simplified_train_test_df.rename(columns={simplified_train_test_df.columns[1]: 'phenotype'}, inplace=True)
		## sort by samples
		simplified_train_test_df = simplified_train_test_df.sort_values(by='sample')

		# check if the output directory does not exists and make it
		if not os.path.exists(OUTPUTPATH):
			os.makedirs(OUTPUTPATH)
			message_output_directory = "The output directory was created successfully"
			print(message_output_directory)
		else:
			message_output_directory = "The output directory already existed"
			print(message_output_directory)

		# step control
		step1_end = dt.datetime.now()
		step1_diff = step1_end - step1_start

		# output results
		## output path
		outpath_count_classes = OUTPUTPATH + '/' + PREFIX + '_count_classes' + '.tsv'
		outpath_features = OUTPUTPATH + '/' + PREFIX + '_features' + '.obj'
		outpath_encoded_features = OUTPUTPATH + '/' + PREFIX + '_encoded_features' + '.obj'
		outpath_encoded_classes = OUTPUTPATH + '/' + PREFIX + '_encoded_classes' + '.obj'
		outpath_model = OUTPUTPATH + '/' + PREFIX + '_model' + '.obj'
		outpath_scores_parameters = OUTPUTPATH + '/' + PREFIX + '_scores_parameters' + '.tsv'
		outpath_cm_classes_train = OUTPUTPATH + '/' + PREFIX + '_confusion_matrix_classes_training' + '.tsv'
		outpath_cm_classes_test = OUTPUTPATH + '/' + PREFIX + '_confusion_matrix_classes_testing' + '.tsv'
		outpath_metrics_classes_train = OUTPUTPATH + '/' + PREFIX + '_metrics_classes_training' + '.tsv'
		outpath_metrics_classes_test = OUTPUTPATH + '/' + PREFIX + '_metrics_classes_testing' + '.tsv'
		outpath_metrics_global_train = OUTPUTPATH + '/' + PREFIX + '_metrics_global_training' + '.tsv'
		outpath_metrics_global_test = OUTPUTPATH + '/' + PREFIX + '_metrics_global_testing' + '.tsv'
		outpath_train = OUTPUTPATH + '/' + PREFIX + '_prediction_training' + '.tsv'
		outpath_test = OUTPUTPATH + '/' + PREFIX + '_prediction_testing' + '.tsv'
		outpath_phenotype_dataset = OUTPUTPATH + '/' + PREFIX + '_phenotype_dataset' + '.tsv'
		outpath_log = OUTPUTPATH + '/' + PREFIX + '_modeling_log' + '.txt'
		## write output in a tsv file
		count_classes_df.to_csv(outpath_count_classes, sep="\t", index=False, header=True)
		cm_classes_train_df.to_csv(outpath_cm_classes_train, sep="\t", index=False, header=True)
		cm_classes_test_df.to_csv(outpath_cm_classes_test, sep="\t", index=False, header=True)
		metrics_classes_train_df.to_csv(outpath_metrics_classes_train, sep="\t", index=False, header=True)
		metrics_classes_test_df.to_csv(outpath_metrics_classes_test, sep="\t", index=False, header=True)
		metrics_global_train_df.to_csv(outpath_metrics_global_train, sep="\t", index=False, header=True)
		metrics_global_test_df.to_csv(outpath_metrics_global_test, sep="\t", index=False, header=True)
		all_scores_parameters_df.to_csv(outpath_scores_parameters, sep="\t", index=False, header=True)
		combined_train_df.to_csv(outpath_train, sep="\t", index=False, header=True)
		combined_test_df.to_csv(outpath_test, sep="\t", index=False, header=True)
		simplified_train_test_df.to_csv(outpath_phenotype_dataset, sep="\t", index=False, header=True)
		## save the features
		with open(outpath_features, 'wb') as file:
			pi.dump(features, file)
		## save the encoded_features
		with open(outpath_encoded_features, 'wb') as file:
			pi.dump(encoded_features, file)
		## save the encoded_classes for the XGB model
		if CLASSIFIER == 'XGB':
			with open(outpath_encoded_classes, 'wb') as file:
				pi.dump(encoded_classes, file)
		## save the model
		with open(outpath_model, 'wb') as file:
			pi.dump(best_model, file)
		## write output in a txt file
		log_file = open(outpath_log, "w")
		log_file.writelines(["########################\n####### context  #######\n########################\n"])
		print(context, file=log_file)
		log_file.writelines(["########################\n###### reference  ######\n########################\n"])
		print(reference, file=log_file)
		log_file.writelines(["########################\n##### repositories #####\n########################\n"])
		print(parser.epilog, file=log_file)
		log_file.writelines(["########################\n### acknowledgements ###\n########################\n"])
		print(acknowledgements, file=log_file)
		log_file.writelines(["########################\n####### versions #######\n########################\n"])
		log_file.writelines("GenomicBasedClassification: " + __version__ + " (released in " + __release__ + ")" + "\n")
		log_file.writelines("python: " + str(sys.version_info[0]) + "." + str(sys.version_info[1]) + "\n")
		log_file.writelines("argparse: " + str(ap.__version__) + "\n")
		log_file.writelines("pickle: " + str(pi.format_version) + "\n")
		log_file.writelines("pandas: " + str(pd.__version__) + "\n")
		log_file.writelines("imblearn: " + str(imb.__version__) + "\n")
		log_file.writelines("sklearn: " + str(sk.__version__) + "\n")
		log_file.writelines("xgboost: " + str(xgb.__version__) + "\n")
		log_file.writelines("numpy: " + str(np.__version__) + "\n")
		log_file.writelines("joblib: " + str(jl.__version__) + "\n")
		log_file.writelines("tqdm: " + str(tq.__version__) + "\n")
		log_file.writelines("tqdm-joblib: " + str(imp.version("tqdm-joblib")) + "\n")
		log_file.writelines(["########################\n####### arguments ######\n########################\n"])
		for key, value in vars(args).items():
			log_file.write(f"{key}: {value}\n")
		log_file.writelines(["########################\n######## samples #######\n########################\n"])
		print(count_classes_df, file=log_file)
		log_file.writelines(["########################\n######## checks ########\n########################\n"])
		log_file.writelines(message_warnings + "\n")
		log_file.writelines(message_traceback + "\n")
		log_file.writelines(message_versions + "\n")
		log_file.writelines(message_number_phenotype_classes + "\n")
		log_file.writelines(message_input_mutations + "\n")
		log_file.writelines(message_input_phenotypes + "\n")
		log_file.writelines(message_missing_phenotypes + "\n")
		log_file.writelines(message_expected_datasets + "\n")
		log_file.writelines(message_sample_identifiers + "\n")
		log_file.writelines(message_class_encoder + "\n")
		log_file.writelines(message_compatibility_dataset_slitting + "\n")
		log_file.writelines(message_dataset + "\n")
		log_file.writelines(message_count_classes + "\n")
		log_file.writelines(message_classifier + "\n")
		if CLASSIFIER == 'XGB':
			log_file.writelines(message_XGB_type_phenotype_classes + "\n")
		log_file.writelines(message_parameters + "\n")
		log_file.writelines(message_best_parameters + "\n")
		log_file.writelines(message_best_score + "\n")
		log_file.writelines(message_output_directory + "\n")
		log_file.writelines(["########################\n###### execution #######\n########################\n"])
		log_file.writelines("The script started on " + str(step1_start) + "\n")
		log_file.writelines("The script stoped on " + str(step1_end) + "\n")
		secs = step1_diff.total_seconds()
		days,secs = divmod(secs,secs_per_day:=60*60*24)
		hrs,secs = divmod(secs,secs_per_hr:=60*60)
		mins,secs = divmod(secs,secs_per_min:=60)
		secs = round(secs, 2)
		message_duration='The script lasted {} days, {} hrs, {} mins and {} secs'.format(int(days),int(hrs),int(mins),secs)
		log_file.writelines(message_duration + "\n")
		log_file.writelines(["########################\n##### output files #####\n########################\n"])
		log_file.writelines(outpath_count_classes + "\n")
		log_file.writelines(outpath_train + "\n")
		log_file.writelines(outpath_test + "\n")
		log_file.writelines(outpath_scores_parameters + "\n")
		log_file.writelines(outpath_features + "\n")
		log_file.writelines(outpath_encoded_features + "\n")
		if CLASSIFIER == 'XGB':
			log_file.writelines(outpath_encoded_classes + "\n")
		log_file.writelines(outpath_model + "\n")
		log_file.writelines(outpath_phenotype_dataset + "\n")
		log_file.writelines(outpath_cm_classes_train + "\n")
		log_file.writelines(outpath_cm_classes_test + "\n")
		log_file.writelines(outpath_metrics_classes_train + "\n")
		log_file.writelines(outpath_metrics_classes_test + "\n")
		log_file.writelines(outpath_metrics_global_train + "\n")
		log_file.writelines(outpath_metrics_global_test + "\n")
		log_file.writelines(outpath_log + "\n")
		log_file.writelines(["########################\n### confusion matrix ###\n########################\n"])
		log_file.writelines(f"from the training dataset: \n")
		print(cm_classes_train_df.to_string(index=False), file=log_file)
		log_file.writelines(f"from the testing dataset: \n")
		print(cm_classes_test_df.to_string(index=False), file=log_file)
		log_file.writelines(f"NB: The expectation and prediction are represented by rows and columns, respectively. \n")
		log_file.writelines(["########################\n### metrics per class ##\n########################\n"])
		log_file.writelines(f"from the training dataset: \n")
		print(metrics_classes_train_df.to_string(index=False), file=log_file)
		log_file.writelines(f"from the testing dataset: \n")
		print(metrics_classes_test_df.to_string(index=False), file=log_file)
		log_file.writelines(f"NB: The term 'support' corresponds to TP + FN. \n")
		log_file.writelines(f"NB: MCC stands for Matthews Correlation Coefficient. \n")
		log_file.writelines(f"NB: Sensitivity and recall must be equal, as they are based on the same formula. \n")
		log_file.writelines(["########################\n#### global metrics ####\n########################\n"])
		log_file.writelines(f"from the training dataset: \n")
		print(metrics_global_train_df.to_string(index=False), file=log_file)
		log_file.writelines(f"from the testing dataset: \n")
		print(metrics_global_test_df.to_string(index=False), file=log_file)
		log_file.writelines(f"NB: MCC stands for Matthews Correlation Coefficient. \n")
		log_file.writelines(f"NB: Sensitivity and recall must be equal, as they are based on the same formula. \n")
		log_file.writelines(["########################\n### training dataset ###\n########################\n"])
		print(combined_train_df.to_string(index=False), file=log_file)
		log_file.writelines(["########################\n### testing  dataset ###\n########################\n"])
		print(combined_test_df.to_string(index=False), file=log_file)
		log_file.close()
		
	elif args.subcommand == 'prediction':
		
		# print a message about subcommand
		message_subcommand = "The prediction subcommand was used"
		print(message_subcommand)

		# read input files
		## mutations
		df_mutations = pd.read_csv(INPUTPATH_MUTATIONS, sep='\t', dtype=str)
		### check the input file of mutations
		#### calculate the number of rows
		rows_mutations = len(df_mutations)
		#### calculate the number of columns
		columns_mutations = len(df_mutations.columns)
		#### check if more than 1 samples rows and 4 columns
		if (rows_mutations >= 1) and (columns_mutations >= 4): 
			message_input_mutations = "The number of expected rows (i.e. >= 1) and columns (i.e. >= 4) of the input file of mutations was properly controled (i.e. " + str(rows_mutations) + " and " + str(columns_mutations) + " , respectively)"
			print (message_input_mutations)
		else: 
			message_input_mutations = "The number of expected rows (i.e. >= 1) and columns (i.e. >= 4) of the input file of mutations was inproperly controled (i.e. " + str(rows_mutations) + " and " + str(columns_mutations) + " , respectively)"
			raise Exception(message_input_mutations)
		## features
		with open(INPUTPATH_FEATURES, 'rb') as file:
			features = pi.load(file)
		## encoded features
		with open(INPUTPATH_ENCODED_FEATURES, 'rb') as file:
			encoded_features = pi.load(file)
		## encoded class for the XGB model
		if INPUTPATH_ENCODED_CLASSES == None:
			message_encoded_classes = "The encoded classes were not provided"
			print(message_encoded_classes)		
		else:
			message_encoded_classes = "The encoded classes were provided"
			print(message_encoded_classes)
			with open(INPUTPATH_ENCODED_CLASSES, 'rb') as file:
				encoded_classes = pi.load(file)
		## model
		with open(INPUTPATH_MODEL, 'rb') as file:
			loaded_model = pi.load(file)

		# detect the loaded model
		detected_model = loaded_model.__class__.__name__
		message_detected_model = "The classifier of the provided best model was properly recognized: " + detected_model
		print(message_detected_model)
		
		# check compatibility between model and encoded classes required for the XGB model
		if (INPUTPATH_ENCODED_CLASSES != None) and (detected_model != "XGBClassifier"):
			message_compatibility_model_classes = "The classifier of the provided best model was not XGB and did not require to provide encoded classes"
			raise Exception(message_compatibility_model_classes)
		elif (INPUTPATH_ENCODED_CLASSES == None) and (detected_model == "XGBClassifier"):
			message_compatibility_model_classes = "The classifier of the provided best model was XGB and required to provide encoded classes"
			raise Exception(message_compatibility_model_classes)
		else:
			message_compatibility_model_classes = "The classifier of the provided best model was verified for compatibility with encoded classes, which are only used for the XGB classifier"
			print(message_compatibility_model_classes)
		
		# replace missing genomic data by a sting
		df_mutations = df_mutations.fillna('missing')
		# rename variables of headers
		df_mutations.rename(columns={df_mutations.columns[0]: 'sample'}, inplace=True)
		# sort by samples
		df_mutations = df_mutations.sort_values(by='sample')
		# prepare mutations indexing the sample columns
		X_mutations = df_mutations.set_index('sample')

		# encode string features into binary features for the dataset to predict
		## instantiate encoder object
		encoder = OneHotEncoder(handle_unknown = 'ignore', sparse_output = False, categories = encoded_features).set_output(transform='pandas')
		# transform data with the encoder into an array for the testing dataset using produced encoded features
		X_mutations_encoded = encoder.fit_transform(X_mutations[features])
		
		# perform prediction
		y_pred_mutations = loaded_model.predict(X_mutations_encoded)

		# prepare output results
		## transform numpy.ndarray into pandas.core.frame.DataFrame
		y_pred_mutations_df = pd.DataFrame(y_pred_mutations)
		## retrieve the sample index in a column
		y_samples_df = pd.DataFrame(X_mutations_encoded.reset_index().iloc[:, 0])
		## concatenate horizontally with reset index
		combined_mutations_df = pd.concat([y_samples_df.reset_index(drop=True), y_pred_mutations_df.reset_index(drop=True)], axis=1)
		## rename variables of headers
		combined_mutations_df.rename(columns={0: 'prediction'}, inplace=True)
		## transform back the phenotype numbers into phenotype classes for the XGB model
		if INPUTPATH_ENCODED_CLASSES != None:
			le = LabelEncoder()
			le.classes_ = encoded_classes
			combined_mutations_df["prediction"] = le.inverse_transform(combined_mutations_df["prediction"])
		
		# retrieve trained phenotype classes
		if INPUTPATH_ENCODED_CLASSES != None:
			classes_lst = encoded_classes
		else:
			classes_lst = loaded_model.classes_

		# retrieve p-values
		## as a numpy.ndarray
		y_pvalues_mutations_nda = loaded_model.predict_proba(X_mutations_encoded)
		## transform numpy.ndarray into pandas.core.frame.DataFrame
		y_pvalues_mutations_df = pd.DataFrame(y_pvalues_mutations_nda, columns = classes_lst)

		# concatenate horizontally the combined predictions and p-values without reset index
		combined_mutations_df = pd.concat([combined_mutations_df, y_pvalues_mutations_df], axis=1)

		# round digits of a dataframe avoid SettingWithCopyWarning with .copy()
		combined_mutations_df = combined_mutations_df.copy()
		combined_mutations_df[combined_mutations_df.select_dtypes(include='number').columns] = combined_mutations_df.select_dtypes(include='number').round(digits)

		# check if the output directory does not exists and make it
		if not os.path.exists(OUTPUTPATH):
			os.makedirs(OUTPUTPATH)
			message_output_directory = "The output directory was created successfully"
			print(message_output_directory)
		else:
			message_output_directory = "The output directory already existed"
			print(message_output_directory)
		
		# step control
		step1_end = dt.datetime.now()
		step1_diff = step1_end - step1_start

		# output results
		## output path
		outpath_prediction = OUTPUTPATH + '/' + PREFIX + '_prediction' + '.tsv'
		outpath_log = OUTPUTPATH + '/' + PREFIX + '_prediction_log' + '.txt'
		## write output in a tsv file
		combined_mutations_df.to_csv(outpath_prediction, sep="\t", index=False, header=True)
		## write output in a txt file
		log_file = open(outpath_log, "w")
		log_file.writelines(["########################\n####### context  #######\n########################\n"])
		print(context, file=log_file)
		log_file.writelines(["########################\n###### reference  ######\n########################\n"])
		print(reference, file=log_file)
		log_file.writelines(["########################\n##### repositories #####\n########################\n"])
		print(parser.epilog, file=log_file)
		log_file.writelines(["########################\n### acknowledgements ###\n########################\n"])
		print(acknowledgements, file=log_file)
		log_file.writelines(["########################\n####### versions #######\n########################\n"])
		log_file.writelines("GenomicBasedClassification: " + __version__ + " (released in " + __release__ + ")" + "\n")
		log_file.writelines("python: " + str(sys.version_info[0]) + "." + str(sys.version_info[1]) + "\n")
		log_file.writelines("argparse: " + str(ap.__version__) + "\n")
		log_file.writelines("pickle: " + str(pi.format_version) + "\n")
		log_file.writelines("pandas: " + str(pd.__version__) + "\n")
		log_file.writelines("imblearn: " + str(imb.__version__) + "\n")
		log_file.writelines("sklearn: " + str(sk.__version__) + "\n")
		log_file.writelines("xgboost: " + str(xgb.__version__) + "\n")
		log_file.writelines("numpy: " + str(np.__version__) + "\n")
		log_file.writelines("joblib: " + str(jl.__version__) + "\n")
		log_file.writelines("tqdm: " + str(tq.__version__) + "\n")
		log_file.writelines("tqdm-joblib: " + str(imp.version("tqdm-joblib")) + "\n")
		log_file.writelines(["########################\n####### arguments ######\n########################\n"])
		for key, value in vars(args).items():
			log_file.write(f"{key}: {value}\n")
		log_file.writelines(["########################\n######## checks ########\n########################\n"])
		log_file.writelines(message_warnings + "\n")
		log_file.writelines(message_traceback + "\n")
		log_file.writelines(message_versions + "\n")
		log_file.writelines(message_input_mutations + "\n")
		log_file.writelines(message_encoded_classes + "\n")
		log_file.writelines(message_detected_model + "\n")
		log_file.writelines(message_compatibility_model_classes + "\n")
		log_file.writelines(message_output_directory + "\n")
		log_file.writelines(["########################\n###### execution #######\n########################\n"])
		log_file.writelines("The script started on " + str(step1_start) + "\n")
		log_file.writelines("The script stoped on " + str(step1_end) + "\n")
		secs = step1_diff.total_seconds()
		days,secs = divmod(secs,secs_per_day:=60*60*24)
		hrs,secs = divmod(secs,secs_per_hr:=60*60)
		mins,secs = divmod(secs,secs_per_min:=60)
		secs = round(secs, 2)
		message_duration='The script lasted {} days, {} hrs, {} mins and {} secs'.format(int(days),int(hrs),int(mins),secs)
		log_file.writelines(message_duration + "\n")
		log_file.writelines(["########################\n##### output files #####\n########################\n"])
		log_file.writelines(outpath_prediction + "\n")
		log_file.writelines(outpath_log + "\n")
		log_file.writelines(["########################\n## prediction dataset ##\n########################\n"])
		print(combined_mutations_df.to_string(index=False), file=log_file)
		log_file.close()
	# print final messages
	print(message_duration)
	print("The results are ready: " + OUTPUTPATH)
	print(parser.epilog)

# identify the block which will only be run when the script is executed directly
if __name__ == "__main__":
	main()
