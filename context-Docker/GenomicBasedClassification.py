# required librairies
## pip3.12 install --force-reinstall pandas==2.2.2
## pip3.12 install --force-reinstall imbalanced-learn==0.13.0
## pip3.12 install --force-reinstall scikit-learn==1.5.2
## pip3.12 install --force-reinstall xgboost==2.1.3
## pip3.12 install --force-reinstall lightgbm==4.6.0
## pip3.12 install --force-reinstall catboost==1.2.8
## pip3.12 install --force-reinstall numpy==1.26.4
## pip3.12 install --force-reinstall joblib==1.5.1
## pip3.12 install --force-reinstall tqdm==4.67.1
## pip3.12 install --force-reinstall tqdm-joblib==0.0.4

'''
# examples of commands
## for the ADA classifier
python3.12 GenomicBasedClassification.py modeling -m genomic_profiles_for_modeling.tsv -ph phenotype_dataset.tsv -o MyDirectory -x ADA_FirstAnalysis -da random -sp 80 -c ADA -k 5 -pa tuning_parameters_ADA.txt -pi -nr 5
python3.12 GenomicBasedClassification.py prediction -m genomic_profiles_for_prediction.tsv -t MyDirectory/ADA_FirstAnalysis_model.obj -f MyDirectory/ADA_FirstAnalysis_features.obj -fe MyDirectory/ADA_FirstAnalysis_feature_encoder.obj -o MyDirectory -x ADA_SecondAnalysis
## for the CAT classifier
python3.12 GenomicBasedClassification.py modeling -m genomic_profiles_for_modeling.tsv -ph MyDirectory/ADA_FirstAnalysis_phenotype_dataset.tsv -o MyDirectory -x CAT_FirstAnalysis -da manual -fs SKB -c CAT -k 5 -pa tuning_parameters_CAT.txt -pi -nr 5
python3.12 GenomicBasedClassification.py prediction -m genomic_profiles_for_prediction.tsv -t MyDirectory/CAT_FirstAnalysis_model.obj -f MyDirectory/CAT_FirstAnalysis_features.obj -fe MyDirectory/CAT_FirstAnalysis_feature_encoder.obj -o MyDirectory -x CAT_SecondAnalysis
## for the DT classifier
python3.12 GenomicBasedClassification.py modeling -m genomic_profiles_for_modeling.tsv -ph MyDirectory/ADA_FirstAnalysis_phenotype_dataset.tsv -o MyDirectory -x DT_FirstAnalysis -da manual -fs laSFM -c DT -k 5 -pa tuning_parameters_DT.txt -pi -nr 5
python3.12 GenomicBasedClassification.py prediction -m genomic_profiles_for_prediction.tsv -t MyDirectory/DT_FirstAnalysis_model.obj -f MyDirectory/DT_FirstAnalysis_features.obj -fe MyDirectory/DT_FirstAnalysis_feature_encoder.obj -o MyDirectory -x DT_SecondAnalysis
## for the EN classifier
python3.12 GenomicBasedClassification.py modeling -m genomic_profiles_for_modeling.tsv -ph MyDirectory/ADA_FirstAnalysis_phenotype_dataset.tsv -o MyDirectory -x EN_FirstAnalysis -da manual -fs enSFM -c EN -k 5 -pa tuning_parameters_EN.txt -pi -nr 5
python3.12 GenomicBasedClassification.py prediction -m genomic_profiles_for_prediction.tsv -t MyDirectory/EN_FirstAnalysis_model.obj -f MyDirectory/EN_FirstAnalysis_features.obj -fe MyDirectory/EN_FirstAnalysis_feature_encoder.obj -o MyDirectory -x EN_SecondAnalysis
## for the ET classifier
python3.12 GenomicBasedClassification.py modeling -m genomic_profiles_for_modeling.tsv -ph MyDirectory/ADA_FirstAnalysis_phenotype_dataset.tsv -o MyDirectory -x ET_FirstAnalysis -da manual -fs riSFM -c ET -k 5 -pa tuning_parameters_ET.txt -pi -nr 5
python3.12 GenomicBasedClassification.py prediction -m genomic_profiles_for_prediction.tsv -t MyDirectory/ET_FirstAnalysis_model.obj -f MyDirectory/ET_FirstAnalysis_features.obj -fe MyDirectory/ET_FirstAnalysis_feature_encoder.obj -o MyDirectory -x ET_SecondAnalysis
## for the GB classifier
python3.12 GenomicBasedClassification.py modeling -m genomic_profiles_for_modeling.tsv -ph MyDirectory/ADA_FirstAnalysis_phenotype_dataset.tsv -o MyDirectory -x GB_FirstAnalysis -da manual -fs rfSFM -c GB -k 5 -pa tuning_parameters_GB.txt -pi -nr 5
python3.12 GenomicBasedClassification.py prediction -m genomic_profiles_for_prediction.tsv -t MyDirectory/GB_FirstAnalysis_model.obj -f MyDirectory/GB_FirstAnalysis_features.obj -fe MyDirectory/GB_FirstAnalysis_feature_encoder.obj -o MyDirectory -x GB_SecondAnalysis
## for the GNB classifier
python3.12 GenomicBasedClassification.py modeling -m genomic_profiles_for_modeling.tsv -ph MyDirectory/ADA_FirstAnalysis_phenotype_dataset.tsv -o MyDirectory -x GNB_FirstAnalysis -da manual -fs SKB -c GNB -k 5 -pa tuning_parameters_GNB.txt -pi -nr 5
python3.12 GenomicBasedClassification.py prediction -m genomic_profiles_for_prediction.tsv -t MyDirectory/GNB_FirstAnalysis_model.obj -f MyDirectory/GNB_FirstAnalysis_features.obj -fe MyDirectory/GNB_FirstAnalysis_feature_encoder.obj -o MyDirectory -x GNB_SecondAnalysis
## for the HGB classifier
python3.12 GenomicBasedClassification.py modeling -m genomic_profiles_for_modeling.tsv -ph MyDirectory/ADA_FirstAnalysis_phenotype_dataset.tsv -o MyDirectory -x HGB_FirstAnalysis -da manual -fs SKB -c HGB -k 5 -pa tuning_parameters_HGB.txt -pi -nr 5
python3.12 GenomicBasedClassification.py prediction -m genomic_profiles_for_prediction.tsv -t MyDirectory/HGB_FirstAnalysis_model.obj -f MyDirectory/HGB_FirstAnalysis_features.obj -fe MyDirectory/HGB_FirstAnalysis_feature_encoder.obj -o MyDirectory -x HGB_SecondAnalysis
## for the KNN classifier
python3.12 GenomicBasedClassification.py modeling -m genomic_profiles_for_modeling.tsv -ph MyDirectory/ADA_FirstAnalysis_phenotype_dataset.tsv -o MyDirectory -x KNN_FirstAnalysis -da manual -fs SKB -c KNN -k 5 -pa tuning_parameters_KNN.txt -pi -nr 5
python3.12 GenomicBasedClassification.py prediction -m genomic_profiles_for_prediction.tsv -t MyDirectory/KNN_FirstAnalysis_model.obj -f MyDirectory/KNN_FirstAnalysis_features.obj -fe MyDirectory/KNN_FirstAnalysis_feature_encoder.obj -o MyDirectory -x KNN_SecondAnalysis
## for the LA classifier
python3.12 GenomicBasedClassification.py modeling -m genomic_profiles_for_modeling.tsv -ph MyDirectory/ADA_FirstAnalysis_phenotype_dataset.tsv -o MyDirectory -x LA_FirstAnalysis -da manual -fs SKB -c LA -k 5 -pa tuning_parameters_LA.txt -pi -nr 5
python3.12 GenomicBasedClassification.py prediction -m genomic_profiles_for_prediction.tsv -t MyDirectory/LA_FirstAnalysis_model.obj -f MyDirectory/LA_FirstAnalysis_features.obj -fe MyDirectory/LA_FirstAnalysis_feature_encoder.obj -o MyDirectory -x LA_SecondAnalysis
## for the LDA classifier
python3.12 GenomicBasedClassification.py modeling -m genomic_profiles_for_modeling.tsv -ph MyDirectory/ADA_FirstAnalysis_phenotype_dataset.tsv -o MyDirectory -x LDA_FirstAnalysis -da manual -fs SKB -c LDA -k 5 -pa tuning_parameters_LDA.txt -pi -nr 5
python3.12 GenomicBasedClassification.py prediction -m genomic_profiles_for_prediction.tsv -t MyDirectory/LDA_FirstAnalysis_model.obj -f MyDirectory/LDA_FirstAnalysis_features.obj -fe MyDirectory/LDA_FirstAnalysis_feature_encoder.obj -o MyDirectory -x LDA_SecondAnalysis
## for the LGBM classifier
python3.12 GenomicBasedClassification.py modeling -m genomic_profiles_for_modeling.tsv -ph MyDirectory/ADA_FirstAnalysis_phenotype_dataset.tsv -o MyDirectory -x LGBM_FirstAnalysis -da manual -fs SKB -c LGBM -k 5 -pa tuning_parameters_LGBM.txt -pi -nr 5
python3.12 GenomicBasedClassification.py prediction -m genomic_profiles_for_prediction.tsv -t MyDirectory/LGBM_FirstAnalysis_model.obj -f MyDirectory/LGBM_FirstAnalysis_features.obj -fe MyDirectory/LGBM_FirstAnalysis_feature_encoder.obj -o MyDirectory -x LGBM_SecondAnalysis
## for the LR classifier
python3.12 GenomicBasedClassification.py modeling -m genomic_profiles_for_modeling.tsv -ph MyDirectory/ADA_FirstAnalysis_phenotype_dataset.tsv -o MyDirectory -x LR_FirstAnalysis -da manual -fs SKB -c LR -k 5 -pa tuning_parameters_LR.txt -pi -nr 5
python3.12 GenomicBasedClassification.py prediction -m genomic_profiles_for_prediction.tsv -t MyDirectory/LR_FirstAnalysis_model.obj -f MyDirectory/LR_FirstAnalysis_features.obj -fe MyDirectory/LR_FirstAnalysis_feature_encoder.obj -o MyDirectory -x LR_SecondAnalysis
## for the MLP classifier
python3.12 GenomicBasedClassification.py modeling -m genomic_profiles_for_modeling.tsv -ph MyDirectory/ADA_FirstAnalysis_phenotype_dataset.tsv -o MyDirectory -x MLP_FirstAnalysis -da manual -fs SKB -c MLP -k 5 -pa tuning_parameters_MLP.txt -pi -nr 5
python3.12 GenomicBasedClassification.py prediction -m genomic_profiles_for_prediction.tsv -t MyDirectory/MLP_FirstAnalysis_model.obj -f MyDirectory/MLP_FirstAnalysis_features.obj -fe MyDirectory/MLP_FirstAnalysis_feature_encoder.obj -o MyDirectory -x MLP_SecondAnalysis
## for the NSV classifier
python3.12 GenomicBasedClassification.py modeling -m genomic_profiles_for_modeling.tsv -ph MyDirectory/ADA_FirstAnalysis_phenotype_dataset.tsv -o MyDirectory -x NSV_FirstAnalysis -da manual -fs SKB -c NSV -k 5 -pa tuning_parameters_NSV.txt -pi -nr 5
python3.12 GenomicBasedClassification.py prediction -m genomic_profiles_for_prediction.tsv -t MyDirectory/NSV_FirstAnalysis_model.obj -f MyDirectory/NSV_FirstAnalysis_features.obj -fe MyDirectory/NSV_FirstAnalysis_feature_encoder.obj -o MyDirectory -x NSV_SecondAnalysis
## for the QDA classifier
python3.12 GenomicBasedClassification.py modeling -m genomic_profiles_for_modeling.tsv -ph MyDirectory/ADA_FirstAnalysis_phenotype_dataset.tsv -o MyDirectory -x QDA_FirstAnalysis -da manual -fs SKB -c QDA -k 5 -pa tuning_parameters_QDA.txt -pi -nr 5
python3.12 GenomicBasedClassification.py prediction -m genomic_profiles_for_prediction.tsv -t MyDirectory/QDA_FirstAnalysis_model.obj -f MyDirectory/QDA_FirstAnalysis_features.obj -fe MyDirectory/QDA_FirstAnalysis_feature_encoder.obj -o MyDirectory -x QDA_SecondAnalysis
## for the RF classifier
python3.12 GenomicBasedClassification.py modeling -m genomic_profiles_for_modeling.tsv -ph MyDirectory/ADA_FirstAnalysis_phenotype_dataset.tsv -o MyDirectory -x RF_FirstAnalysis -da manual -fs SKB -c RF -k 5 -pa tuning_parameters_RF.txt -pi -nr 5
python3.12 GenomicBasedClassification.py prediction -m genomic_profiles_for_prediction.tsv -t MyDirectory/RF_FirstAnalysis_model.obj -f MyDirectory/RF_FirstAnalysis_features.obj -fe MyDirectory/RF_FirstAnalysis_feature_encoder.obj -o MyDirectory -x RF_SecondAnalysis
## for the RI classifier
python3.12 GenomicBasedClassification.py modeling -m genomic_profiles_for_modeling.tsv -ph MyDirectory/ADA_FirstAnalysis_phenotype_dataset.tsv -o MyDirectory -x RI_FirstAnalysis -da manual -fs SKB -c RI -k 5 -pa tuning_parameters_RI.txt -pi -nr 5
python3.12 GenomicBasedClassification.py prediction -m genomic_profiles_for_prediction.tsv -t MyDirectory/RI_FirstAnalysis_model.obj -f MyDirectory/RI_FirstAnalysis_features.obj -fe MyDirectory/RI_FirstAnalysis_feature_encoder.obj -o MyDirectory -x RI_SecondAnalysis
## for the SVC classifier
python3.12 GenomicBasedClassification.py modeling -m genomic_profiles_for_modeling.tsv -ph MyDirectory/ADA_FirstAnalysis_phenotype_dataset.tsv -o MyDirectory -x SVC_FirstAnalysis -da manual -fs SKB -c SVC -k 5 -pa tuning_parameters_SVC.txt -pi -nr 5
python3.12 GenomicBasedClassification.py prediction -m genomic_profiles_for_prediction.tsv -t MyDirectory/SVC_FirstAnalysis_model.obj -f MyDirectory/SVC_FirstAnalysis_features.obj -fe MyDirectory/SVC_FirstAnalysis_feature_encoder.obj -o MyDirectory -x SVC_SecondAnalysis
## for the XGB classifier
python3.12 GenomicBasedClassification.py modeling -m genomic_profiles_for_modeling.tsv -ph MyDirectory/ADA_FirstAnalysis_phenotype_dataset.tsv -o MyDirectory -x XGB_FirstAnalysis -da manual -fs SKB -c XGB -k 5 -pa tuning_parameters_XGB.txt -pi -nr 5
python3.12 GenomicBasedClassification.py prediction -m genomic_profiles_for_prediction.tsv -t MyDirectory/XGB_FirstAnalysis_model.obj -f MyDirectory/XGB_FirstAnalysis_features.obj -fe MyDirectory/XGB_FirstAnalysis_feature_encoder.obj -ce MyDirectory/XGB_FirstAnalysis_class_encoder.obj -o MyDirectory -x XGB_SecondAnalysis
'''
# import packages
## standard libraries
import sys as sys # no individual installation because is part of the Python Standard Library (no version)
import os as os # no individual installation because is part of the Python Standard Library (no version)
import datetime as dt # no individual installation because is part of the Python Standard Library (no version)
import argparse as ap # no individual installation because is part of the Python Standard Library
import pickle as pi # no individual installation because is part of the Python Standard Library
import warnings as wa # no individual installation because is part of the Python Standard Library (no version)
import re as re # no individual installation because is part of the Python Standard Library (with version)
import importlib.metadata as imp # no individual installation because is part of the Python Standard Library (no version)
import functools as ft # no individual installation because is part of the Python Standard Library (no version)
import contextlib as ctl # no individual installation because is part of the Python Standard Library (no version)
import io as io # no individual installation because is part of the Python Standard Library (no version)
import threadpoolctl as tpc # no individual installation because is part of the Python Standard Library (no version)

## third-party libraries
import pandas as pd
import imblearn as imb
import sklearn as sk
import xgboost as xgb
import lightgbm as lgbm
import numpy as np
import joblib as jl
import tqdm as tq
import tqdm.auto as tqa # no version because it corresponds a tqdm module
import tqdm_joblib as tqjl
import catboost as cb
from sklearn import set_config
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, label_binarize
from sklearn.model_selection import train_test_split, GridSearchCV, ParameterGrid
from sklearn.metrics import confusion_matrix, multilabel_confusion_matrix, classification_report, accuracy_score, precision_score, recall_score, f1_score, cohen_kappa_score, matthews_corrcoef, roc_auc_score, average_precision_score, precision_recall_curve
from sklearn.svm import SVC, NuSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, HistGradientBoostingClassifier, ExtraTreesClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.inspection import permutation_importance
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis, LinearDiscriminantAnalysis
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, chi2, f_regression, mutual_info_classif, SelectFromModel
from catboost import CatBoostClassifier, Pool

# compatibility patch: prevent GridSearchCV from injecting random_state into CatBoost
class SafeCatBoostClassifier(CatBoostClassifier):
	"""a subclass of CatBoostClassifier that safely ignores sklearn random_state parameter."""
	def set_params(self, **params):
		# Drop sklearn’s automatic random_state injection to avoid CatBoostError
		params.pop("random_state", None)
		return super().set_params(**params)

# set static metadata to keep outside the main function
## set workflow repositories
repositories = 'Please cite:\n GitHub (https://github.com/Nicolas-Radomski/GenomicBasedClassification),\n Docker Hub (https://hub.docker.com/r/nicolasradomski/genomicbasedclassification),\n and/or Anaconda Hub (https://anaconda.org/nicolasradomski/genomicbasedclassification).'
## set the workflow context
context = "The scikit-learn (sklearn)-based Python workflow is inspired by an older caret-based R workflow (https://doi.org/10.1186/s12864-023-09667-w), independently supports both modeling (i.e., training and testing) and prediction (i.e., based on a pre-built model), and implements 5 feature selection methods, 20 model classifiers, hyperparameter tuning, performance metric computation, feature and permutation importance analyses, prediction probability estimation, execution monitoring via progress bars, and parallel processing."
## set the initial workflow reference
reference = "Pierluigi Castelli, Andrea De Ruvo, Andrea Bucciacchio, Nicola D'Alterio, Cesare Camma, Adriano Di Pasquale and Nicolas Radomski (2023) Harmonization of supervised machine learning practices for efficient source attribution of Listeria monocytogenes based on genomic data. 2023, BMC Genomics, 24(560):1-19, https://doi.org/10.1186/s12864-023-09667-w"
## set the acknowledgement
acknowledgements = "Many thanks to Andrea De Ruvo, Adriano Di Pasquale and ChatGPT for the insightful discussions that helped improve the algorithm."
# set the version and release
__version__ = "1.3.0"
__release__ = "December 2025"

# set global sklearn config early
set_config(transform_output="pandas")

# define functions of interest

def compute_curve_metrics(y_true, y_proba_nda, classes, digits, eps=1e-8):
	"""
	compute per-class ROC-AUC, PR-AUC, PRG-AUC and PRG-AUC_clipped for multi-class or binary predictions.
	parameters
	----------
	y_true : array-like, shape (n_samples,)
		True class labels (can be strings or integers).
	y_proba_nda : array-like, shape (n_samples, n_classes)
		Predicted probabilities for each class.
		Must match the order of 'classes' columns.
	classes : list of length n_classes
		Class labels corresponding to columns of y_proba_nda.
	digits : int
		Number of decimal places to round the metrics.
	eps : float, default=1e-8
		Small value to clip probabilities to avoid constant arrays.
	returns
	-------
	metrics_df : pd.DataFrame, shape (n_classes, 5)
		DataFrame with columns:
		['phenotype', 'ROC-AUC', 'PR-AUC', 'PRG-AUC', 'PRG-AUC_clipped'].
	"""
	# binarize labels for multi-class
	# for binary classification, label_binarize returns a single column by default
	y_true_bin_nda = label_binarize(y_true, classes=classes)
	# fix for binary classification
	# if there are exactly 2 classes and only one column was created,
	# add the complement column to match the expected 2-column format
	if y_true_bin_nda.shape[1] == 1 and len(classes) == 2:
		y_true_bin_nda = np.hstack([1 - y_true_bin_nda, y_true_bin_nda])
	# initialize results DataFrame
	metrics_df = pd.DataFrame(columns=['phenotype', 'ROC-AUC', 'PR-AUC', 'PRG-AUC', 'PRG-AUC_clipped'])
	# loop over each class to compute class-dependent curve-based metrics
	for i, cls in enumerate(classes):
		# extract binary true labels for this class
		y_binary = y_true_bin_nda[:, i]
		# extract predicted probabilities for this class
		# clip probabilities to avoid exact 0 or 1 which may cause errors in metrics
		y_scores = np.clip(y_proba_nda[:, i], eps, 1 - eps)
		# compute ROC-AUC
		try:
			roc_auc = roc_auc_score(y_binary, y_scores)
		except ValueError:
			roc_auc = 0.0
		# compute PR-AUC (Average Precision)
		try:
			pr_auc = average_precision_score(y_binary, y_scores)
		except ValueError:
			pr_auc = 0.0
		# compute raw PRG-AUC (can be negative, zero, or positive)
		try:
			prg_auc, _, _, _ = compute_prg_auc(y_binary, y_scores)
		except Exception:
			prg_auc = 0.0
		# compute clipped PRG-AUC (negative precision gains floored at 0)
		try:
			prg_auc_clipped, _, _, _ = compute_prg_auc(y_binary, y_scores, clip_negative=True)
		except Exception:
			prg_auc_clipped = 0.0
		# store rounded results in the DataFrame
		metrics_df.loc[i] = [
			cls,
			round(roc_auc, digits),
			round(pr_auc, digits),
			round(prg_auc, digits),
			round(prg_auc_clipped, digits)
		]
	# return final per-class ROC-AUC, PR-AUC, PRG-AUC and PRG-AUC_clipped DataFrame
	return metrics_df

def compute_prg_auc(y_true, y_scores, eps=1e-12, clip_negative=False):
	"""
	compute PRG curve and AUC using Flach & Kull (2015) formulation.
	parameters
	----------
	y_true : array-like, shape (n_samples,)
		True binary class labels (0 for negative, 1 for positive).
	y_scores : array-like, shape (n_samples,)
		Predicted probabilities for the positive class.
	eps : float, default=1e-12
		Small value added to avoid division by zero during gain computation.
	clip_negative : bool, default=False
		If True, negative precision gain values are clipped to 0 before integration.
	returns
	-------
	auc_prg : float
		Area under the Precision-Recall-Gain curve.
	precision_gain : ndarray
		Array of precision gain values at each threshold.
	recall_gain : ndarray
		Array of recall gain values at each threshold.
	thresholds : ndarray
		Decision thresholds corresponding to each precision-recall point.
	"""
	# proportion of positive samples (π)
	pi = np.mean(y_true)
	# compute standard precision-recall curve
	precision, recall, thresholds = precision_recall_curve(y_true, y_scores)
	# avoid division by zero in precision
	precision = np.maximum(precision, eps)
	# compute precision gain and recall gain
	precision_gain = (precision - pi) / (1 - pi)
	recall_gain = 1 - (1 - recall) / (1 - pi)
	# numerical safety: replace NaN/Inf and clamp to [0,1]
	precision_gain = np.nan_to_num(precision_gain, nan=0.0, posinf=0.0, neginf=0.0)
	recall_gain = np.nan_to_num(recall_gain, nan=0.0, posinf=0.0, neginf=0.0)
	precision_gain = np.clip(precision_gain, 0.0, 1.0)
	recall_gain = np.clip(recall_gain, 0.0, 1.0)
	# optionally clip negative precision gains to 0
	if clip_negative:
		precision_gain = np.maximum(precision_gain, 0)
	# ensure recall_gain is in ascending order before integration
	if recall_gain[0] > recall_gain[-1]:
		recall_gain = recall_gain[::-1]
		precision_gain = precision_gain[::-1]
	# integrate PRG curve using trapezoidal rule
	auc_prg = np.trapz(precision_gain, recall_gain)
	return auc_prg, precision_gain, recall_gain, thresholds

def count_selected_features(pipeline, encoded_matrix):
	"""
	robust count of features the pipeline expects
	returns the number of columns reaching the final estimator
	handles both Pipeline objects and direct estimators
	"""
	# ensure the object is a pipeline; wrap standalone estimators
	if not hasattr(pipeline, "named_steps"):
		pipeline = Pipeline([("model", pipeline)])
	# check if a feature selection step exists
	if "feature_selection" in pipeline.named_steps:
		fs = pipeline.named_steps["feature_selection"]
		# support_ is the most reliable indicator
		if hasattr(fs, "support_") and fs.support_ is not None:
			return int(np.sum(fs.support_))
		# convert to a NumPy float32 array (CAT issue)
		try:
			X_np = np.asarray(encoded_matrix, dtype=np.float32)
		except Exception:
			X_np = encoded_matrix.values.astype(np.float32)
		# try transform on a single sample
		try:
			return fs.transform(X_np[:1]).shape[1]
		except Exception:
			pass
		# fallback: try full transform
		try:
			return fs.transform(X_np).shape[1]
		except Exception:
			pass
		# last resort: selector exists but unusable → full width
		return int(X_np.shape[1])
	# no explicit selector → check the estimator directly
	est = pipeline.named_steps.get("model", pipeline)
	n_feat = getattr(est, "n_features_in_", None)
	# sklearn 1.3+ compatibility
	if n_feat is None and hasattr(est, "feature_names_in_"):
		n_feat = len(est.feature_names_in_)
	# CatBoost, XGB, HGB often hide n_features_in_
	if n_feat is None or n_feat == 0:
		n_feat = encoded_matrix.shape[1]
	return int(n_feat)

def restricted_float_split(x: str) -> float:
	"""
	convert *x* to float and ensure 0 < x < 100
	raises
	------
	argparse.ArgumentTypeError
		if *x* cannot be parsed as float or is not in (0, 100)
	"""
	try:
		x = float(x)
	except ValueError:
		raise ap.ArgumentTypeError(f"{x!r} is not a valid float")
	if not (0.0 < x < 100.0):
		raise ap.ArgumentTypeError("split must be a float in the open interval (0, 100)")
	return x

def restricted_int_limit(x: str) -> int:
	"""
	convert *x* to int and ensure x >= 1
	raises
	------
	argparse.ArgumentTypeError
		if *x* cannot be parsed as int or is less than 1
	"""
	try:
		x = int(x)
	except ValueError:
		raise ap.ArgumentTypeError(f"{x!r} is not a valid integer")
	if x < 1:
		raise ap.ArgumentTypeError("limit must be an integer ≥ 1")
	return x

def restricted_int_fold(x: str) -> int:
	"""
	convert *x* to int and ensure x ≥ 2
	raises
	------
	argparse.ArgumentTypeError
		if *x* cannot be parsed as int or is less than 2
	"""
	try:
		x = int(x)
	except ValueError:
		raise ap.ArgumentTypeError(f"{x!r} is not a valid integer")
	if x < 2:
		raise ap.ArgumentTypeError("fold must be an integer ≥ 2 for cross-validation")
	return x

def restricted_int_jobs(x: str) -> int:
	"""
	convert *x* to int and ensure x == -1 or x ≥ 1
	raises
	------
	argparse.ArgumentTypeError
		if *x* cannot be parsed as int or is not -1 or ≥ 1
	"""
	try:
		x = int(x)
	except ValueError:
		raise ap.ArgumentTypeError(f"{x!r} is not a valid integer")
	if x != -1 and x < 1:
		raise ap.ArgumentTypeError("jobs must be -1 (all CPUs) or an integer ≥ 1")
	return x

def restricted_int_nrepeats(x: str) -> int:
	"""
	convert *x* to int and ensure x ≥ 1
	raises
	------
	argparse.ArgumentTypeError
		if *x* cannot be parsed as int or is less than 1
	"""
	try:
		x = int(x)
	except ValueError:
		raise ap.ArgumentTypeError(f"{x!r} is not a valid integer")
	if x < 1:
		raise ap.ArgumentTypeError("nrepeats must be an integer ≥ 1 for permutation importance")
	return x

def restricted_int_digits(x: str) -> int:
	"""
	convert *x* to int and ensure x ≥ 0
	raises
	------
	argparse.ArgumentTypeError
		if *x* cannot be parsed as int or is negative.
	"""
	try:
		x = int(x)
	except ValueError:
		raise ap.ArgumentTypeError(f"{x!r} is not a valid integer")
	if x < 0:
		raise ap.ArgumentTypeError("digits must be an integer ≥ 0")
	return x

def restricted_debug_level(x: str) -> int:
	"""
	convert *x* to int and ensure x >= 0.
	raises
	------
	argparse.ArgumentTypeError
		if *x* cannot be parsed as int or is negative.
	"""
	try:
		x = int(x)
	except ValueError:
		raise ap.ArgumentTypeError(f"{x!r} is not a valid integer")
	if x < 0:
		raise ap.ArgumentTypeError("debug must be zero or a positive integer (0, 1, 2, ...)")
	return x

# create a main function preventing the global scope from being unintentionally executed on import
def main():

	# step control
	step1_start = dt.datetime.now()

	# create the main parser
	parser = ap.ArgumentParser(
		prog="GenomicBasedClassification.py", 
		description="Perform classification-based modeling or prediction from binary (e.g., presence/absence of genes) or categorical (e.g., allele profiles) genomic data.",
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
		help='Absolute or relative input path of tab-separated values (tsv) file including profiles of mutations. First column: sample identifiers identical to those in the input file of phenotypes and datasets (header: e.g., sample). Other columns: profiles of mutations (header: labels of mutations). [MANDATORY]'
		)
	parser_modeling.add_argument(
		'-ph', '--phenotypes', 
		dest='inputpath_phenotypes', 
		action='store', 
		required=True, 
		help="Absolute or relative input path of tab-separated values (tsv) file including profiles of phenotypes and datasets. First column: sample identifiers identical to those in the input file of mutations (header: e.g., sample). Second column: categorical phenotype (header: e.g., phenotype). Third column: 'training' or 'testing' dataset (header: e.g., dataset). [MANDATORY]"
		)
	parser_modeling.add_argument(
		'-da', '--dataset', 
		dest='dataset', 
		type=str,
		action='store', 
		required=False, 
		choices=['random', 'manual'], 
		default='random', 
		help="Perform random (i.e., 'random') or manual (i.e., 'manual') splitting of training and testing datasets through the holdout method. [OPTIONAL, DEFAULT: 'random']"
		)
	parser_modeling.add_argument(
		'-sp', '--split', 
		dest='splitting', 
		type=restricted_float_split, # control (0, 100) open interval
		action='store', 
		required=False, 
		default=None, 
		help='Percentage of random splitting to prepare the training dataset through the holdout method. [OPTIONAL, DEFAULT: None]'
		)
	parser_modeling.add_argument(
		'-l', '--limit', 
		dest='limit', 
		type=restricted_int_limit, # control >= 1
		action='store', 
		required=False, 
		default=10, 
		help='Recommended minimum of samples per class in both the training and testing datasets to reliably estimate performance metrics. [OPTIONAL, DEFAULT: 10]'
		)
	parser_modeling.add_argument(
		'-fs', '--featureselection', 
		dest='featureselection', 
		type=str,
		action='store', 
		required=False, 
		default='None', 
		help='Acronym of the classification-compatible feature selection method to use: SelectKBest (SKB), SelectFromModel with lasso-regularized logistic regression (laSFM), SelectFromModel with elasticNet-regularized logistic regression (enSFM), SelectFromModel with ridge-regularized logistic regression (riSFM), or SelectFromModel with random forest (rfSFM). These methods are suitable for high-dimensional binary or categorical-encoded features. [OPTIONAL, DEFAULT: None]'
		)
	parser_modeling.add_argument(
		'-c', '--classifier', 
		dest='classifier', 
		type=str,
		action='store', 
		required=False, 
		default='XGB', 
		help='Acronym of the classifier to use among adaboost (ADA), catboost (CAT), decision tree (DT), elasticnet-regularized logistic regression (EN), extra trees (ET), gradient boosting (GB), gaussian naive bayes (GNB), histogram-based gradient boosting (HGB), k-nearest neighbors (KNN), lasso-regularized logistic regression (LA), linear discriminant analysis (LDA), light gradient goosting machine (LGBM), logistic regression (LR), multi-layer perceptron (MLP), nu support vector (NSV), quadratic discriminant analysis (QDA), random forest (RF), ridge-regularized logistic regression (RI), support vector classification (SVC) or extreme gradient boosting (XGB). [OPTIONAL, DEFAULT: XGB]'
		)
	parser_modeling.add_argument(
		'-k', '--fold', 
		dest='fold', 
		type=restricted_int_fold, # control >= 2
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
		type=restricted_int_jobs, # control -1 or >= 1
		action='store', 
		required=False, 
		default=-1, 
		help='Value defining the number of jobs to run in parallel compatible with the n_jobs argument of the GridSearchCV function. [OPTIONAL, DEFAULT: -1]'
		)
	parser_modeling.add_argument(
		'-pi', '--permutationimportance', 
		dest='permutationimportance', 
		action='store_true', 
		required=False, 
		default=False, 
		help='Compute permutation importance, which can be computationally expensive, especially with many features and/or high repetition counts. [OPTIONAL, DEFAULT: False]'
		)
	parser_modeling.add_argument(
		'-nr', '--nrepeats', 
		dest='nrepeats', 
		type=restricted_int_nrepeats, # control >= 1
		action='store', 
		required=False, 
		default=10, 
		help='Number of repetitions per feature for permutation importance; higher values provide more stable estimates but increase runtime. [OPTIONAL, DEFAULT: 10]'
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
		'-di', '--digits', 
		dest='digits', 
		type=restricted_int_digits, # control >= 0
		action='store', 
		required=False, 
		default=6, 
		help='Number of decimal digits to round numerical results (e.g., accuracy, importance, metrics). [OPTIONAL, DEFAULT: 6]'
		)
	parser_modeling.add_argument(
		'-de', '--debug', 
		dest='debug', 
		type=restricted_debug_level, # control >= 0
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
		help='Absolute or relative input path of a tab-separated values (tsv) file including profiles of mutations. First column: sample identifiers identical to those in the input file of phenotypes and datasets (header: e.g., sample). Other columns: profiles of mutations (header: labels of mutations). [MANDATORY]'
		)
	parser_prediction.add_argument(
		'-f', '--features', 
		dest='inputpath_features', 
		action='store', 
		required=True, 
		help='Absolute or relative input path of an object (obj) file including features from the training dataset (i.e., mutations). [MANDATORY]'
		)
	parser_prediction.add_argument(
		'-fe', '--featureencoder', 
		dest='inputpath_feature_encoder', 
		action='store', 
		required=True, 
		help='Absolute or relative input path of an object (obj) file including encoder from the training dataset (i.e., mutations). [MANDATORY]'
		)
	parser_prediction.add_argument(
		'-t', '--model', 
		dest='inputpath_model', 
		action='store', 
		required=True, 
		help='Absolute or relative input path of an object (obj) file including a trained scikit-learn model. [MANDATORY]'
		)
	parser_prediction.add_argument(
		'-ce', '--classencoder', 
		dest='inputpath_class_encoder', 
		action='store', 
		required=False, 
		help='Absolute or relative input path of an object (obj) file including trained scikit-learn class encoder (i.e., phenotypes) for the XGB model. [OPTIONAL]'
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
		'-di', '--digits', 
		dest='digits', 
		type=restricted_int_digits, # control >= 0
		action='store', 
		required=False, 
		default=6, 
		help='Number of decimal digits to round numerical results (e.g., accuracy, importance, metrics). [OPTIONAL, DEFAULT: 6]'
		)
	parser_prediction.add_argument(
		'-de', '--debug', 
		dest='debug', 
		type=restricted_debug_level, # control >= 0
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

	# reshape arguments
	## parse the arguments
	args = parser.parse_args()
	## rename arguments
	if args.subcommand == 'modeling':
		INPUTPATH_MUTATIONS=args.inputpath_mutations
		INPUTPATH_PHENOTYPES=args.inputpath_phenotypes
		OUTPUTPATH=args.outputpath
		DATASET=args.dataset
		SPLITTING=args.splitting
		LIMIT=args.limit
		FEATURESELECTION = args.featureselection
		CLASSIFIER=args.classifier
		FOLD=args.fold
		PARAMETERS=args.parameters
		JOBS=args.jobs
		PERMUTATIONIMPORTANCE=args.permutationimportance
		NREPEATS=args.nrepeats
		PREFIX=args.prefix
		DIGITS=args.digits
		DEBUG=args.debug
		WARNINGS=args.warnings
		NOCHECK=args.nocheck
	elif args.subcommand == 'prediction':
		INPUTPATH_MUTATIONS=args.inputpath_mutations
		INPUTPATH_FEATURES=args.inputpath_features
		INPUTPATH_FEATURE_ENCODER=args.inputpath_feature_encoder
		INPUTPATH_CLASS_ENCODER=args.inputpath_class_encoder
		INPUTPATH_MODEL=args.inputpath_model
		OUTPUTPATH=args.outputpath
		PREFIX=args.prefix
		DIGITS=args.digits
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
		if jl.__version__ != "1.5.1":
			raise Exception("joblib 1.5.1 version is recommended")
		if lgbm.__version__ != "4.6.0":
			raise Exception("lightgbm 4.6.0 version is recommended")
		if tq.__version__ != "4.67.1":
			raise Exception("tqdm 4.67.1 version is recommended")
		if imp.version("tqdm-joblib") != "0.0.4":
			raise Exception("tqdm-joblib 0.0.4 version is recommended")
		if imp.version("catboost") != "1.2.8":
			raise Exception("catboost 1.2.8 version is recommended")
		message_versions = "The recommended versions of Python and packages were properly controlled"
	elif NOCHECK == True :
		message_versions = "The recommended versions of Python and packages were not controlled"

	# print a message about version control
	print(message_versions)

	# set rounded digits
	digits = DIGITS

	# check the subcommand and execute corresponding code
	if args.subcommand == 'modeling':

		# print a message about subcommand
		message_subcommand = "The modeling subcommand was used"
		print(message_subcommand)

		# manage minimal limits of samples
		if LIMIT < 10:
			message_limit = (
				"The provided sample limit per class and dataset (i.e., " + str(LIMIT) + ") was below the recommended minimum (i.e., 10) and may lead to unreliable performance metrics"
			)
			print(message_limit)
		else: 
			message_limit = (
				"The provided sample limit per class and dataset (i.e., " + str(LIMIT) + ") meets or exceeds the recommended minimum (i.e., 10), which is expected to support more reliable performance metrics"
			)
			print(message_limit)

		# read input files
		## mutations
		df_mutations = pd.read_csv(INPUTPATH_MUTATIONS, sep='\t', dtype=str)
		## phenotypes
		df_phenotypes = pd.read_csv(INPUTPATH_PHENOTYPES, sep='\t', dtype=str)

		# indentify the type of phenotype classes
		## make sure that the phenotype is provided in the second column
		if df_phenotypes.shape[1] < 2:
			message_number_phenotype_classes = "The presence of phenotype in the input file of phenotypes was improperly controlled (i.e., the second column is missing)"
			raise Exception(message_number_phenotype_classes)
		## count the phenotype classes
		### count each phenotype classes
		counts_each_classes_series = df_phenotypes.groupby(df_phenotypes.columns[1]).size()
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

		# define minimal limites of samples (i.e., 2 * counts_classes_int * LIMIT per class)
		limit_samples = 2 * counts_classes_int * LIMIT

		# check the input file of mutations
		## calculate the number of rows
		rows_mutations = len(df_mutations)
		## calculate the number of columns
		columns_mutations = len(df_mutations.columns)
		## check if more than limit_samples rows and 3 columns
		if (rows_mutations >= limit_samples) and (columns_mutations >= 3): 
			message_input_mutations = "The minimum required number of samples in the training/testing datasets (i.e., >= " + str(limit_samples) + ") and the expected number of columns (i.e., >= 3) in the input file of mutations were properly controlled (i.e., " + str(rows_mutations) + " and " + str(columns_mutations) + " , respectively)"
			print (message_input_mutations)
		else: 
			message_input_mutations = "The minimum required number of samples in the training/testing datasets (i.e., >= " + str(limit_samples) + ") and the expected number of columns (i.e., >= 3) in the input file of mutations were not properly controlled (i.e., " + str(rows_mutations) + " and " + str(columns_mutations) + " , respectively)"
			raise Exception(message_input_mutations)

		# check the input file of phenotypes
		## calculate the number of rows
		rows_phenotypes = len(df_phenotypes)
		## calculate the number of columns
		columns_phenotypes = len(df_phenotypes.columns)
		## check if more than limit_samples rows and 3 columns
		if (rows_phenotypes >= limit_samples) and (columns_phenotypes == 3): 
			message_input_phenotypes = "The minimum required number of samples in the training/testing datasets (i.e., >= " + str(limit_samples) + ") and the expected number of columns (i.e., = 3) in the input file of phenotypes were properly controlled (i.e., " + str(rows_phenotypes) + " and " + str(columns_phenotypes) + " , respectively)"
			print (message_input_phenotypes)
		else: 
			message_input_phenotypes = "The minimum required number of samples in the training/testing datasets (i.e., >= " + str(limit_samples) + ") and the expected number of columns (i.e., = 3) in the input file of phenotypes were not properly controlled (i.e., " + str(rows_phenotypes) + " and " + str(columns_phenotypes) + " , respectively)"
			raise Exception(message_input_phenotypes)
		## check the absence of missing data in the second column (i.e., phenotype)
		missing_phenotypes = pd.Series(df_phenotypes.iloc[:,1]).isnull().values.any()
		if missing_phenotypes == False: 
			message_missing_phenotypes = "The absence of missing phenotypes in the input file of phenotypes was properly controled (i.e., the second column)"
			print (message_missing_phenotypes)
		elif missing_phenotypes == True:
			message_missing_phenotypes = "The absence of missing phenotypes in the input file of phenotypes was inproperly controled (i.e., the second column)"
			raise Exception(message_missing_phenotypes)
		## check the absence of values other than 'training' or 'testing' in the third column (i.e., dataset)
		if (DATASET == "manual"):
			expected_datasets = all(df_phenotypes.iloc[:,2].isin(["training", "testing"]))
			if expected_datasets == True: 
				message_expected_datasets = "The expected datasets (i.e., 'training' or 'testing') in the input file of phenotypes were properly controled (i.e., the third column)"
				print (message_expected_datasets)
			elif expected_datasets == False:
				message_expected_datasets = "The expected datasets (i.e., 'training' or 'testing') in the input file of phenotypes were inproperly controled (i.e., the third column)"
				raise Exception(message_expected_datasets)
		elif (DATASET == "random"):
			message_expected_datasets = "The expected datasets (i.e., 'training' or 'testing') in the input file of phenotypes were not controled (i.e., the third column)"
			print (message_expected_datasets)

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
			class_encoder = LabelEncoder()
			df_phenotypes["phenotype"] = class_encoder.fit_transform(df_phenotypes["phenotype"])
			encoded_classes = class_encoder.classes_
			message_class_encoder = "The phenotype classes were encoded for the XGB classifier (i.e., 0, 1, 2 ....): " + str(", ".join(f"'{item}'" for item in encoded_classes))
			print(message_class_encoder)
		else:
			message_class_encoder = "The phenotype classes were not encoded for the classifiers other than the XGB classifier"
			print(message_class_encoder)

		# check compatibility between the dataset and splitting arguments
		if (DATASET == 'random') and (SPLITTING != None):
			message_compatibility_dataset_slitting = "The provided selection of training/testing datasets (i.e., " + DATASET + ") and percentage of random splitting (i.e., " + str(SPLITTING) + "%) were compatible"
			print(message_compatibility_dataset_slitting)
		elif (DATASET == 'random') and (SPLITTING == None):
			message_compatibility_dataset_slitting = "The provided selection of training/testing datasets (i.e., " + DATASET + ") required the percentage of random splitting (i.e., " + str(SPLITTING) + ")"
			raise Exception(message_compatibility_dataset_slitting)
		elif (DATASET == 'manual') and (SPLITTING == None):
			message_compatibility_dataset_slitting = "The provided selection of training/testing datasets (i.e., " + DATASET + ") and percentage of random splitting (i.e., " + str(SPLITTING) + ") were compatible"
			print(message_compatibility_dataset_slitting)
		elif (DATASET == 'manual') and (SPLITTING != None):
			message_compatibility_dataset_slitting = "The provided selection of training/testing datasets (i.e., " + DATASET + ") did not require the percentage of random splitting (i.e., " + str(SPLITTING) + "%)"
			raise Exception(message_compatibility_dataset_slitting)

		# perform splitting of the training and testing datasets according to the setting
		if DATASET == 'random':
			message_dataset = "The training and testing datasets were constructed based on the 'random' setting"
			print(message_dataset)
			# drop dataset column (since it's not needed)
			df_phenotypes = df_phenotypes.drop("dataset", axis='columns')
			# merge phenotypes and mutations deterministically
			df_all = (
				pd.merge(df_phenotypes, df_mutations, on="sample", how="inner")
				.sort_values(by="sample")
				.reset_index(drop=True)
			)
			# do not normalize df_all["dataset"] here because it does not exist for random split
			# create the dataframes mutations (X) and phenotypes (y)
			X = df_all.drop(columns=["phenotype"])  # keep "sample" for now
			y = df_all[["sample", "phenotype"]]     # include sample column explicitly
			# index with sample identifiers
			X.set_index("sample", inplace=True)
			y.set_index("sample", inplace=True)
			# split the dataset into training and testing sets (without random_state=42 to avoid reproducibility)
			X_train, X_test, y_train, y_test = train_test_split(
				X, y, stratify=y, train_size=SPLITTING / 100
			)
			# convert y_train and y_test into clean 1-D Series indexed by sample
			y_train = y_train["phenotype"]
			y_test  = y_test ["phenotype"]
		elif DATASET == 'manual':
			message_dataset = "The training and testing datasets were constructed based on the 'manual' setting"
			print(message_dataset)
			# merge phenotypes and mutations deterministically
			df_all = (
				pd.merge(df_phenotypes, df_mutations, on="sample", how="inner")
				.sort_values(by="sample")
				.reset_index(drop=True)
			)
			# normalize only here (since dataset column exists)
			df_all["dataset"] = df_all["dataset"].astype(str).str.strip().str.lower()
			# split according to dataset column
			df_training = df_all[df_all["dataset"] == "training"]
			df_testing  = df_all[df_all["dataset"] == "testing"]
			# build X and y dataframes for training/testing
			## extract numerical genomic features and set sample identifiers as index
			X_train = df_training.drop(columns=["phenotype", "dataset"]).set_index("sample")
			X_test  = df_testing .drop(columns=["phenotype", "dataset"]).set_index("sample")
			## extract phenotype as a clean 1-D Series indexed by sample
			y_train = df_training[["sample", "phenotype"]].set_index("sample")["phenotype"]
			y_test  = df_testing [ ["sample", "phenotype"]].set_index("sample")["phenotype"]

		# check number of samples per class
		## retrieve a list of unique classes
		### transform a dataframe column into a list
		classes_unique_lst = df_phenotypes['phenotype'].tolist()
		### remove replicates
		classes_unique_lst = list(set(classes_unique_lst))
		### sort by alphabetic order
		classes_unique_lst.sort()
		## count classes
		### in the whole dataset
		count_dataset_lst = df_phenotypes['phenotype'].value_counts().reindex(classes_unique_lst, fill_value=0).tolist()
		### in the training dataset
		#### ensure y_train behaves as 1-D phenotype Series
		if isinstance(y_train, pd.DataFrame):
			y_train_series = y_train["phenotype"]
		else:
			y_train_series = y_train
		count_train_lst = (
			y_train_series
			.value_counts()
			.reindex(classes_unique_lst, fill_value=0)
			.tolist()
		)
		### in the testing dataset
		#### ensure y_test behaves as 1-D phenotype Series
		if isinstance(y_test, pd.DataFrame):
			y_test_series = y_test["phenotype"]
		else:
			y_test_series = y_test
		count_test_lst = (
			y_test_series
			.value_counts()
			.reindex(classes_unique_lst, fill_value=0)
			.tolist()
		)
		## combine horizontally lists into a dataframe
		count_classes_df = pd.DataFrame({
			'phenotype': classes_unique_lst,
			'dataset': count_dataset_lst,
			'training': count_train_lst,
			'testing': count_test_lst
			})
		## transform back the phenotype numbers into phenotype classes for the XGB model
		if CLASSIFIER == 'XGB':
			count_classes_df["phenotype"] = class_encoder.inverse_transform(count_classes_df["phenotype"])
		## control minimal number of samples per class
		### detect small number of samples in the training dataset
		detection_train = all(element >= LIMIT for element in count_classes_df['training'].tolist())
		### detect small number of samples in the testing dataset
		detection_test = all(element >= LIMIT for element in count_classes_df['testing'].tolist())
		### check the minimal quantity of samples per class
		if (detection_train == True) and (detection_test == True):
			message_count_classes = "The number of samples per class in the training and testing datasets was properly controlled to be higher than the set limit (i.e., " + str(LIMIT) + ")"
			print(message_count_classes)
		elif (detection_train == False) or (detection_test == False):
			message_count_classes = "The number of samples per class in the training and testing datasets was improperly controlled, making it lower than the set limit (i.e., " + str(LIMIT) + ")"
			print(count_classes_df.to_string(index=False))
			raise Exception(message_count_classes)

		# encode categorical data into binary data using the one-hot encoder
		## save input feature names from training dataset
		features = X_train.columns
		## instantiate the encoder
		feature_encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False).set_output(transform='pandas')
		## fit encoder on training data (implicitly includes all training features)
		X_train_encoded = feature_encoder.fit_transform(X_train.astype(str))
		## check missing features
		missing_features = set(features) - set(X_test.columns)
		if missing_features:
			message_missing_features = "The following training features expected by the one-hot encoder are missing in the input tested mutations: " + str(missing_features)
			raise Exception(message_missing_features)
		else: 
			message_missing_features = "The input tested mutations include all features required by the trained one-hot encoder"
			print (message_missing_features)
		## check extra features
		extra_features = set(X_test.columns) - set(features)
		if extra_features:
			message_extra_features = "The following unexpected features in the input tested mutations will be ignored for one-hot encoding: " + str(extra_features)
			print (message_extra_features)
		else: 
			message_extra_features = "The input tested mutations contain no unexpected features for one-hot encoding"
			print (message_extra_features)
		## ensure order and cast to str for encoding
		X_test_features_str = X_test[features].astype(str)
		## use the same encoder (already fitted) to transform test data
		X_test_encoded = feature_encoder.transform(X_test_features_str)
		## assert identical encoded features between training and testing datasets
		train_encoded_features = list(X_train_encoded.columns)
		test_encoded_features = list(X_test_encoded.columns)
		if train_encoded_features != test_encoded_features:
			message_assert_encoded_features = "The encoded features between training and testing datasets do not match"
			raise AssertionError(message_assert_encoded_features)
		else:
			message_assert_encoded_features = "The encoded features between training and testing datasets were confirmed as identical"
			print(message_assert_encoded_features)

		# enforce consistent one-hot encoded column order between train/test and across runs
		X_train_encoded = X_train_encoded.reindex(sorted(X_train_encoded.columns), axis=1)
		X_test_encoded  = X_test_encoded.reindex(sorted(X_train_encoded.columns), axis=1, fill_value=0)
		message_column_order = ("The one-hot encoded column order was harmonized across training and testing datasets to ensure deterministic feature alignment for feature selection and modeling")
		print(message_column_order)

		# count features
		## count the number of raw categorical features before one-hot encoding
		features_before_ohe_int = len(features)
		## count the number of binary features after one-hot encoding
		features_after_ohe_int = X_train_encoded.shape[1]
		## print a message
		message_ohe_features = "The " + str(features_before_ohe_int) + " provided features were one-hot encoded into " + str(features_after_ohe_int) + " encoded features"
		print(message_ohe_features)

		# prepare elements of the model
		## initialize the feature selection method (without tuning parameters: deterministic randomness (random_state=42) without repeatability for parallel nondeterminism (n_jobs=1 or thread_count=1))
		if FEATURESELECTION == 'None':
			message_feature_selection = "The provided feature selection method was properly recognized: None"
			print(message_feature_selection)
			selected_feature_selector = None
		elif FEATURESELECTION == 'SKB':
			message_feature_selection = "The provided feature selection method was properly recognized: SelectKBest (SKB)"
			print(message_feature_selection)
			selected_feature_selector = SelectKBest(
				score_func=ft.partial( # partial allow reproducibility
					mutual_info_classif, # available score_func options include mutual_info_classif (default), chi2, and f_regression. The latter are provided for advanced users but may not be suitable for categorical or binary targets.
					random_state=42  # deterministic
				), 
				k=10 # default top k features can be modified in the parameters file if needed
			)
		elif FEATURESELECTION == 'laSFM':
			message_feature_selection = "The provided feature selection method was properly recognized: SelectFromModel with lasso-regularized logistic regression (laSFM)"
			print(message_feature_selection)
			# minimal deterministic setup that still produces sparsity
			selector_model = LogisticRegression(penalty='l1', solver='liblinear', random_state=42)
			selected_feature_selector = SelectFromModel(estimator=selector_model, threshold=None)
		elif FEATURESELECTION == 'enSFM':
			message_feature_selection = "The provided feature selection method was properly recognized: SelectFromModel with elasticnet-regularized logistic regression (enSFM)"
			print(message_feature_selection)
			# minimal deterministic setup; without these, model would be invalid
			selector_model = LogisticRegression(penalty='elasticnet', solver='saga', l1_ratio=0.5, random_state=42)
			selected_feature_selector = SelectFromModel(estimator=selector_model, threshold=None)
		elif FEATURESELECTION == 'riSFM':
			message_feature_selection = "The provided feature selection method was properly recognized: SelectFromModel with ridge-regularized logistic regression (riSFM)"
			print(message_feature_selection)
			# minimal deterministic setup; without these, model would be invalid
			selector_model = LogisticRegression(penalty='l2', solver='lbfgs', random_state=42)
			selected_feature_selector = SelectFromModel(estimator=selector_model, threshold=None)
		elif FEATURESELECTION == 'rfSFM':
			message_feature_selection = "The provided feature selection method was properly recognized: SelectFromModel with random forest (rfSFM)"
			print(message_feature_selection)
			selector_model = RandomForestClassifier(
				random_state=42, # deterministic
				bootstrap=False # disable bootstrapping to reduce random variability in feature importance computation
			)
			selected_feature_selector = SelectFromModel(estimator=selector_model, threshold=None)
		else:
			message_feature_selection = "The provided feature selection method is not implemented yet"
			raise Exception(message_feature_selection)
		## initialize the classifier (without tuning parameters: deterministic randomness (random_state=42) without repeatability for parallel nondeterminism (n_jobs=1 or thread_count=1))
		if CLASSIFIER == 'ADA':
			message_classifier = "The provided classifier was properly recognized: adaboost (ADA)"
			print(message_classifier)
			selected_classifier = AdaBoostClassifier(random_state=42) # adaboost (ADA)
		elif CLASSIFIER == 'CAT':
			message_classifier = "The provided classifier was properly recognized: CatBoost (CAT)"
			print(message_classifier)
			# deterministic configuration for CatBoost
			# - use random_seed (not random_state) to prevent synonym conflicts
			# - allow_writing_files=False disables CatBoost automatic file outputs
			# - bootstrap_type='Bayesian' + random_strength=0 ensure reproducible splits
			# - verbose=False keeps logs clean and avoids stdout clutter
			if type_phenotype_classes == 'more than two classes':
				selected_classifier = SafeCatBoostClassifier(
					loss_function="MultiClass", # default loss for classification of multi classes
					random_seed=42, # deterministic
					verbose=False, # no stdout spam
					allow_writing_files=False, # prevent auto logging files
					bootstrap_type="Bayesian", # deterministic bootstrap
					random_strength=0 # must be 0 for reproducibility
				) # catboost (CAT)
				message_CAT_type_phenotype_classes = ("The CAT classifier was set to manage more than two phenotype classes")
				print(message_CAT_type_phenotype_classes)
			elif type_phenotype_classes == 'two classes':
				selected_classifier = SafeCatBoostClassifier(
					loss_function="Logloss", # default loss for classification of two classes
					random_seed=42, # deterministic
					verbose=False, # no stdout spam
					allow_writing_files=False, # prevent auto logging files
					bootstrap_type="Bayesian", # deterministic bootstrap
					random_strength=0 # must be 0 for reproducibility
				)
				message_CAT_type_phenotype_classes = ("The CAT classifier was set to manage two phenotype classes")
				print(message_CAT_type_phenotype_classes)
		elif CLASSIFIER == 'DT':
			message_classifier = "The provided classifier was properly recognized: decision tree classifier (DT)"
			print(message_classifier)
			selected_classifier = DecisionTreeClassifier(random_state=42) # decision tree (DT)
		elif CLASSIFIER == 'EN':
			message_classifier = "The provided classifier was properly recognized: elasticNet-regularized logistic regression (EN)"
			print(message_classifier)
			# a mix of L1/L2 penalities controlled by l1_ratio corresponds to elasticnet logistic regression, solver saga is required
			selected_classifier = LogisticRegression(penalty='elasticnet', solver='saga', l1_ratio=0.5, random_state=42) # elasticnet-regularized logistic regression (EN)
		elif CLASSIFIER == 'ET':
			message_classifier = "The provided classifier was properly recognized: extra trees classifier (ET)"
			print(message_classifier)
			selected_classifier = ExtraTreesClassifier(random_state=42) # extra trees (ET)
		elif CLASSIFIER == 'GB':
			message_classifier = "The provided classifier was properly recognized: gradient boosting (GB)"
			print(message_classifier)
			selected_classifier = GradientBoostingClassifier(random_state=42)
		elif CLASSIFIER == 'GNB':
			message_classifier = "The provided classifier was properly recognized: gaussian naive bayes (GNB)"
			print(message_classifier)
			selected_classifier = GaussianNB() # gaussian naive bayes (GNB)
		elif CLASSIFIER == 'HGB':
			message_classifier = "The provided classifier was properly recognized: histogram-based gradient boosting (HGB)"
			print(message_classifier)
			selected_classifier = HistGradientBoostingClassifier(random_state=42) # histogram-based gradient boosting (HGB)
		elif CLASSIFIER == 'KNN':
			message_classifier = "The provided classifier was properly recognized: k-nearest neighbors (KNN)"
			print(message_classifier)
			selected_classifier = KNeighborsClassifier() # k-nearest neighbors (KNN)
		elif CLASSIFIER == 'LA':
			message_classifier = "The provided classifier was properly recognized: lasso-regularized logistic regression (LA)"
			print(message_classifier)
			# L1 penalty corresponds to lasso logistic regression, solver liblinear is required
			selected_classifier = LogisticRegression(penalty='l1', solver='liblinear', random_state=42) # lasso-regularized logistic regression (LA)
		elif CLASSIFIER == 'LGBM':
			message_classifier = "The provided classifier was properly recognized: Light Gradient Boosting Machine (LGBM)"
			print(message_classifier)
			selected_classifier = lgbm.LGBMClassifier(random_state=42, verbose=-1)
		elif CLASSIFIER == 'LDA':
			message_classifier = "The provided classifier was properly recognized: linear discriminant analysis (LDA)"
			print(message_classifier)
			selected_classifier = LinearDiscriminantAnalysis() # linear discriminant analysis (LDA)
		elif CLASSIFIER == 'LR':
			message_classifier = "The provided classifier was properly recognized: logistic regression (LR)"
			print(message_classifier)
			selected_classifier = LogisticRegression(random_state=42) # logistic regression (LR)
		elif CLASSIFIER == 'MLP':
			message_classifier = "The provided classifier was properly recognized: multi-layer perceptron (MLP)"
			print(message_classifier)
			selected_classifier = MLPClassifier(random_state=42) # multi-layer perceptron (MLP)
		elif CLASSIFIER == 'NSV':
			message_classifier = "The provided classifier was properly recognized: nu support vector (NSV)"
			print(message_classifier)
			selected_classifier = NuSVC(probability=True)
		elif CLASSIFIER == 'QDA':
			message_classifier = "The provided classifier was properly recognized: quadratic discriminant analysis (QDA)"
			print(message_classifier)
			# minimal deterministic setup with light regularization for numerical stability
			selected_classifier = QuadraticDiscriminantAnalysis(reg_param=0.05, tol=1e-4) # quadratic discriminant analysis (QDA)
		elif CLASSIFIER == 'RF':
			message_classifier = "The provided classifier was properly recognized: random forest (RF)"
			print(message_classifier)
			selected_classifier = RandomForestClassifier(random_state=42) # random forest (RF)
		elif CLASSIFIER == 'RI':
			message_classifier = "The provided classifier was properly recognized: ridge-regularized logistic regression (RI)"
			print(message_classifier)
			# L2 penalty corresponds to ridge logistic regression, solver lbfgs is required
			selected_classifier = LogisticRegression(penalty='l2', solver='lbfgs', random_state=42) # ridge-regularized logistic regression
		elif CLASSIFIER == 'SVC':
			message_classifier = "The provided classifier was properly recognized: support vector classification (SVC)"
			print(message_classifier)
			selected_classifier = SVC(probability = True, random_state=42) # support vector classification (SVC)
		elif CLASSIFIER == 'XGB':
			message_classifier = "The provided classifier was properly recognized: extreme gradient boosting (XGB)"
			print(message_classifier)
			if type_phenotype_classes == 'more than two classes':
				selected_classifier = xgb.XGBClassifier(objective='multi:softprob', random_state=42) # extreme gradient boosting (XGB)
				message_XGB_type_phenotype_classes = "The XGB classifier was set to manage " + type_phenotype_classes + " phenotype classes"
				print(message_XGB_type_phenotype_classes)
			elif type_phenotype_classes == 'two classes':
				selected_classifier = xgb.XGBClassifier(objective = 'binary:logistic', random_state=42) # extreme gradient boosting (XGB)
				message_XGB_type_phenotype_classes = "The XGB classifier was set to manage " + type_phenotype_classes + " phenotype classes"
				print(message_XGB_type_phenotype_classes)
		else:
			message_classifier = "The provided classifier is not implemented yet"
			raise Exception(message_classifier)

		## build the pipeline
		### create an empty list
		steps = []
		### add feature selection step if specified
		if FEATURESELECTION in ['SKB', 'laSFM', 'enSFM', 'riSFM', 'rfSFM']:
			steps.append(('feature_selection', selected_feature_selector))
		### add the final model
		steps.append(('model', selected_classifier))
		### create the pipeline
		selected_pipeline = Pipeline(steps)
		### print a message
		message_pipeline = ("The pipeline components were properly recognized: " + re.sub(r'\s+', ' ', str(selected_pipeline)).strip())
		print(message_pipeline)

		## initialize the parameters
		### if the tuning parameters are not provided by the user
		if PARAMETERS == None:
			parameters = [{}]
			message_parameters = "The tuning parameters were not provided by the user"
			print(message_parameters)
		### if the tuning parameters are provided by the user
		elif PARAMETERS != None:
			### open the provided file of tuning parameters
			parameters_file = open(PARAMETERS, "r")
			### read provided tuning parameters and convert string dictionary to dictionary
			parameters = [eval(parameters_file.read())]
			### close the provided file of tuning parameters
			parameters_file.close()
			### print a message
			message_parameters = "The tuning parameters were provided by the user: " + str(parameters).replace("[", "").replace("]", "")
			print(message_parameters)
		
		# build the model
		## prepare the grid search cross-validation (CV) first
		model = GridSearchCV(
			estimator=selected_pipeline,
			param_grid=parameters,
			cv=FOLD,
			scoring='accuracy',
			verbose=0, # do not display any messages or logs
			n_jobs=JOBS
		)
		## compute metrics related to grid search CV
		### number of distinct parameter names (i.e., how many parameters are tuned)
		n_param_names = len({k for d in parameters for k in d})
		### number of parameter value options (i.e., how many values are tried in total)
		n_total_values = sum(len(v) for d in parameters for v in d.values())
		### number of parameter combinations (i.e., Cartesian product of all value options)
		param_combinations = len(list(ParameterGrid(parameters)))
		### number of fits during cross-validation (i.e., combinations × folds)
		gridsearchcv_fits = param_combinations * FOLD
		### print a message
		message_metrics_cv = "The cross-validation setting implied: " + str(n_param_names) + " distinct parameter names, " + str(n_total_values) + " parameter value options, " + str(param_combinations) + " parameter combinations, and " + str(gridsearchcv_fits) + " fits during cross-validation"
		print(message_metrics_cv)

		# ensure numeric compatibility (astype(np.float32)) with upstream encoding (sparse_output=False) and efficiency (float32 dtype), 
		# especially for tree-based classifiers (e.g., DecisionTreeClassifier, RandomForestClassifier, XGBClassifier, HistGradientBoostingClassifier)
		X_train_encoded_float32 = X_train_encoded.astype(np.float32)
		X_test_encoded_float32 = X_test_encoded.astype(np.float32)
		# preserve sample IDs and enforce 1-D target vectors
		train_sample_ids = y_train.index
		test_sample_ids  = y_test.index
		# reconstruct clean 1-D Series required by CatBoost and Pandas concat
		y_train_series = pd.Series(np.ravel(y_train.values), index=train_sample_ids)
		y_test_series  = pd.Series(np.ravel(y_test.values), index=test_sample_ids)

		# check parallelization and print a message
		if JOBS == 1:
			message_parallelization = "The tqdm_joblib progress bars are deactivated when using one job"
			print(message_parallelization)
		else:
			message_parallelization = "The tqdm_joblib progress bars are activated when using two or more jobs"
			print(message_parallelization)

		## fit the model
		### use tqdm.auto rather than tqdm library because it automatically choose the best display format (terminal, notebook, etc.)
		### use a tqdm progress bar from the tqdm_joblib library (compatible with GridSearchCV)
		### use a tqdm progress bar immediately after the last print (position=0), disable the additional bar after completion (leave=False), and allow for dynamic resizing (dynamic_ncols=True)
		### force GridSearchCV to use the threading backend to avoid the DeprecationWarning from fork and ChildProcessError from the loky backend (default in joblib)
		### threading is slower than loky, but it allows using a progress bar with GridSearchCV and avoids the DeprecationWarning and ChildProcessError
		if JOBS == 1:
			# when using a single thread, tqdm_joblib does not display intermediate progress updates
			# in this case, we run GridSearchCV normally without the tqdm_joblib wrapper
			with jl.parallel_backend('threading', n_jobs=JOBS):
				model.fit(
					X_train_encoded_float32,
					y_train_series
				)
		else:
			# when using multiple threads, tqdm_joblib correctly hooks into joblib and displays progress updates
			with tqa.tqdm(total=gridsearchcv_fits, desc="Model building progress", position=0, leave=False, dynamic_ncols=True) as progress_bar:
				with jl.parallel_backend('threading', n_jobs=JOBS):
					with tqjl.tqdm_joblib(progress_bar):
						model.fit(
							X_train_encoded_float32,
							y_train_series
						)

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
		all_scores_parameters_df = pd.concat(
			[allscores_df, 
			allparameters_df], 
			axis=1, join="inner" # safeguards against accidental row misalignment down the line
		)
		## remove unnecessary characters
		### replace each dictionary by string
		all_scores_parameters_df['parameters'] = all_scores_parameters_df['parameters'].apply(lambda x: str(x))
		### replace special characters { and } by nothing
		all_scores_parameters_df['parameters'] = all_scores_parameters_df['parameters'].replace(r'[\{\}]', '', regex=True)
		## sort the dataframe by scores in descending order and reset the index
		all_scores_parameters_df = all_scores_parameters_df.sort_values(by="scores", ascending=False).reset_index(drop=True)

		# select the best model
		best_model = model.best_estimator_

		# re-fit only the final estimator (not the entire pipeline)
		# this is necessary because re-fitting the whole pipeline would overwrite the
		# feature selection mask learned during GridSearchCV
		# keeping the original mask ensures consistent selected features between
		# modeling and prediction subcommands
		## check if a feature selection step exists inside the pipeline
		if hasattr(best_model, "named_steps") and ("feature_selection" in best_model.named_steps):
			# retrieve feature selector and final estimator
			selector = best_model.named_steps["feature_selection"]
			final_estimator = best_model.named_steps["model"]
			# apply the selector to the harmonized one-hot encoded training matrix
			# use transform() to preserve the exact support_ learned during GridSearchCV
			try:
				X_train_selected = selector.transform(X_train_encoded)
			except Exception:
				# fallback: selector may require float32
				X_train_selected = selector.transform(X_train_encoded_float32)
			# re-fit only the final estimator on selected features
			final_estimator.fit(X_train_selected, y_train_series)
			# rebuild the pipeline WITHOUT re-fitting the selector
			best_model = Pipeline([
				("feature_selection", selector),
				("model", final_estimator)
			])
		else:
			# no feature selection method was used
			# we directly retrieve the final estimator
			if hasattr(best_model, "named_steps"):
				final_estimator = best_model.named_steps["model"]
			else:
				final_estimator = best_model
			# re-fit the final estimator on the full harmonized one-hot encoded training matrix
			# (no feature selection → the estimator expects full encoded matrices)
			final_estimator.fit(X_train_encoded, y_train_series)
			# rebuild the pipeline to maintain a consistent structure
			best_model = Pipeline([
				("model", final_estimator)
			])
		## refresh final estimator reference (required before feature importances and permutation importance)
		final_estimator = (
			best_model.named_steps["model"]
			if hasattr(best_model, "named_steps")
			else best_model
		)

		# count features
		## count the number of selected one-hot encoded features
		selected_features_int = count_selected_features(best_model, X_train_encoded)
		## print a message
		message_selected_features = (
			"The pipeline potentially selected and used "
			+ str(selected_features_int)
			+ " one-hot encoded features to train the model"
		)
		print(message_selected_features)

		# output a dataframe of features used by the final model with ranked importance scores
		## recover encoded feature names before feature selection
		if hasattr(best_model, "named_steps"):
			if "encoder" in best_model.named_steps and hasattr(
				best_model.named_steps["encoder"], "get_feature_names_out"
			):
				encoded_names = best_model.named_steps["encoder"].get_feature_names_out()
			else:
				encoded_names = X_train_encoded.columns
		else:
			encoded_names = X_train_encoded.columns
		## always convert to a clean Python list
		if hasattr(encoded_names, "tolist"):
			feature_encoded_lst_full = encoded_names.tolist()
		else:
			feature_encoded_lst_full = list(encoded_names)
		## determine feature selection mask
		if hasattr(best_model, "named_steps") and "feature_selection" in best_model.named_steps:
			selector = best_model.named_steps["feature_selection"]
			if hasattr(selector, "get_support"):
				support_mask = selector.get_support()
			else:
				support_mask = np.ones(len(feature_encoded_lst_full), dtype=bool)
		else:
			support_mask = np.ones(len(feature_encoded_lst_full), dtype=bool)
		## filtered list for feature importance reporting (not for permutation importance)
		feature_encoded_lst_filtered = np.array(feature_encoded_lst_full)[support_mask].tolist()
		## print a message
		message_importance_encoded_feature_names = (
			"The one-hot encoded feature names were recovered before feature selection"
		)
		print(message_importance_encoded_feature_names)
		## extract feature importance depending on classifier type
		try:
			# catboost must come first (otherwise feature_importances_ steals the branch)
			if isinstance(final_estimator, cb.CatBoostClassifier):
				# CatBoost exposes feature importance via get_feature_importance()
				try:
					train_pool = cb.Pool(X_train_encoded_float32, y_train_series)
					importances = final_estimator.get_feature_importance(
						train_pool, type='PredictionValuesChange'
					)
					importance_type = "CatBoost PredictionValuesChange (feature_importance)"
				except Exception:
					importances = np.array([np.nan] * len(feature_encoded_lst_filtered))
					importance_type = "NaN placeholder (CatBoost importance extraction failed)"
			# LightGBM classifier: supports gain/split feature importance
			elif isinstance(final_estimator, lgbm.LGBMClassifier):
				try:
					# LightGBM supports importance types such as 'gain', 'split', etc.
					lgbm_importance_type = None
					# from user parameters (if present)
					if 'params' in locals():
						lgbm_importance_type = params.get('model__importance_type', None)
					# from estimator internal parameters
					if lgbm_importance_type is None and hasattr(final_estimator, 'get_params'):
						lgbm_importance_type = final_estimator.get_params().get('importance_type', None)
					# default importance type
					if lgbm_importance_type is None:
						lgbm_importance_type = 'gain'

					# LightGBM stores the low-level booster inside .booster_
					importances = final_estimator.booster_.feature_importance(
						importance_type=lgbm_importance_type
					)
					importance_type = f"lightgbm's {lgbm_importance_type}-based importance"
				except Exception:
					# fallback if extraction fails
					importances = np.array([np.nan] * len(feature_encoded_lst_filtered))
					importance_type = "NaN placeholder (LightGBM importance extraction failed)"
			# tree-based classifiers (RF, ET, DT, ADA, XGB exposes feature_importances_)
			elif hasattr(final_estimator, 'feature_importances_'):
				importances = final_estimator.feature_importances_
				importance_type = "tree-based impurity reduction (feature_importances_)"
			# histogram-based GBDT (HGB)
			elif isinstance(final_estimator, HistGradientBoostingClassifier):
				try:
					from sklearn.ensemble._hist_gradient_boosting.utils import get_feature_importances
					importances = get_feature_importances(final_estimator, normalize=True)
				except Exception:
					importances = np.array([])
				if importances is None or len(importances) == 0 or np.all(importances == 0):
					n_features = X_train_encoded.shape[1]
					importances = np.zeros(n_features, dtype=np.float64)
					for predictors in getattr(final_estimator, "_predictors", []):
						if predictors is None:
							continue
						for tree in np.atleast_1d(predictors):
							if hasattr(tree, "split_features_") and hasattr(tree, "split_gains_"):
								for feat, gain in zip(tree.split_features_, tree.split_gains_):
									if feat >= 0:
										importances[int(feat)] += gain
					total_gain = np.sum(importances)
					if total_gain > 0:
						importances /= total_gain
				importance_type = "histogram-based mean impurity reduction (auto fallback to internal split gains)"
			# quadratic discriminant analysis (QDA)
			elif isinstance(final_estimator, QuadraticDiscriminantAnalysis):
				means = final_estimator.means_
				if means.ndim == 2:
					global_mean = np.mean(means, axis=0)
					importances = np.sum((means - global_mean) ** 2, axis=0)
					if np.sum(importances) > 0:
						importances /= np.sum(importances)
					importance_type = "variance-normalized between-class mean difference (derived importance)"
				else:
					importances = np.array([np.nan] * len(feature_encoded_lst_filtered))
					importance_type = "NaN placeholder (QDA means_ missing or invalid)"
			# support vector classification (SVC
			elif isinstance(final_estimator, SVC):
				kernel_type = getattr(final_estimator, 'kernel', 'unknown')
				if kernel_type == 'linear' and hasattr(final_estimator, 'coef_') and not final_estimator.probability:
					importances = np.abs(final_estimator.coef_.ravel())
					importance_type = "absolute coefficient magnitude (linear SVC coef_)"
				else:
					importances = np.array([np.nan] * len(feature_encoded_lst_filtered))
					importance_type = (
						"NaN placeholder (no native importance for " + kernel_type +
						" kernel SVC with probability=" + str(final_estimator.probability) + ")"
					)
			# linear models (LR, LDA, etc.)
			elif hasattr(final_estimator, 'coef_'):
				importances = np.abs(final_estimator.coef_.ravel())
				importance_type = "absolute coefficient magnitude (coef_)"

			# default
			else:
				importances = np.array([np.nan] * len(feature_encoded_lst_filtered))
				importance_type = "NaN placeholder (no native importance)"
		except Exception as e:
			importances = np.array([np.nan] * len(feature_encoded_lst_filtered))
			importance_type = "NaN fallback due to extraction error: " + str(e)
		## handle mismatch between feature names and importances
		if len(importances) != len(feature_encoded_lst_filtered):
			min_len = min(len(importances), len(feature_encoded_lst_filtered))
			importances = importances[:min_len]
			feature_encoded_lst_filtered = feature_encoded_lst_filtered[:min_len]
		## print message about extracted importances
		if CLASSIFIER in ('KNN', 'MLP', 'GNB'):
			message_importance_count = (
				"The selected classifier does not expose feature importances natively ("
				+ importance_type + ")"
			)
		else:
			message_importance_count = (
				"The best model returned " + str(len(importances)) +
				" importance values (" + importance_type + ") for " +
				str(len(feature_encoded_lst_filtered)) + " one-hot encoded features"
			)
		print(message_importance_count)
		## ensure importances array is valid
		if importances is None or np.all(np.isnan(importances)):
			importances = np.full(len(feature_encoded_lst_filtered), np.nan, dtype=float)
		## create feature importance dataframe
		feature_importance_df = pd.DataFrame({
			"feature": feature_encoded_lst_filtered,
			"importance": np.round(importances, digits)
		}).sort_values(by="importance", ascending=False).reset_index(drop=True)

		# check compatibility between permutation importance and the number of repetitions
		set_nrepeats = ('-nr' in sys.argv) or ('--nrepeats' in sys.argv)
		if (PERMUTATIONIMPORTANCE is True) and (set_nrepeats is True):
			message_compatibility_permutation_nrepeat = (
				"The permutation importance was requested (i.e., " + str(PERMUTATIONIMPORTANCE) +
				") and the number of repetitions was explicitly set (i.e., " + str(set_nrepeats) +
				") to a specific value (i.e., " + str(NREPEATS) + ")"
			)
			print(message_compatibility_permutation_nrepeat)
		elif (PERMUTATIONIMPORTANCE is True) and (set_nrepeats is False):
			message_compatibility_permutation_nrepeat = (
				"The permutation importance was requested (i.e., " + str(PERMUTATIONIMPORTANCE) +
				") but the number of repetitions was not set (i.e., " + str(set_nrepeats) +
				"); the default value is therefore used (i.e., " + str(NREPEATS) + ")"
			)
			print(message_compatibility_permutation_nrepeat)
		elif (PERMUTATIONIMPORTANCE is False) and (set_nrepeats is True):
			message_compatibility_permutation_nrepeat = (
				"The permutation importance was not requested (i.e., " + str(PERMUTATIONIMPORTANCE) +
				") but the number of repetitions was set (i.e., " + str(set_nrepeats) +
				"); this setting is consequently ignored (i.e., " + str(NREPEATS) + ")"
			)
			print(message_compatibility_permutation_nrepeat)
		elif (PERMUTATIONIMPORTANCE is False) and (set_nrepeats is False):
			message_compatibility_permutation_nrepeat = (
				"The permutation importance was not requested (i.e., " + str(PERMUTATIONIMPORTANCE) +
				") and the number of repetitions was not set, as expected (i.e., " + str(set_nrepeats) + ")"
			)
			print(message_compatibility_permutation_nrepeat)

		# fix nested parallelism issues for random forest and extra trees so tqdm_joblib stays accurate
		if CLASSIFIER in ['RF', 'ET']:
			try:
				best_model.set_params(model__n_jobs=1)
			except Exception:
				if hasattr(best_model, "n_jobs"):
					best_model.set_params(n_jobs=1)

		# compute permutation importance only if explicitly requested
		# use tqdm.auto rather than tqdm library because it automatically chooses the best display format (terminal, notebook, etc.)
		# use a tqdm progress bar from the tqdm_joblib library (compatible with permutation_importance using joblib parallelism)
		# use threading backend to avoid DeprecationWarning from fork and ChildProcessError from the loky backend
		# threading is slightly slower but ensures smooth tqdm display and avoids nested multiprocessing issues
		if PERMUTATIONIMPORTANCE is True:
			# determine which features to use for permutation importance (selected features only)
			try:
				if hasattr(best_model, "named_steps") and 'feature_selection' in best_model.named_steps:
					selector = best_model.named_steps['feature_selection']
					if hasattr(selector, "get_support"):
						# retrieve boolean mask of selected features
						support_mask = selector.get_support()
						# apply mask directly (do not reorder) on float32-encoded data
						X_train_perm_df = X_train_encoded_float32.loc[:, support_mask]
						X_test_perm_df  = X_test_encoded_float32.loc[:, support_mask]
						# the correct feature names come from the encoder, same order as fit()
						feature_encoded_perm_lst = np.array(feature_encoded_lst_full)[support_mask].tolist()
						# print a message
						message_perm_selection = (
							"The permutation importance was restricted to the features selected upstream (i.e., "
							+ str(len(feature_encoded_perm_lst))
							+ ") by the specified feature selection method"
						)
					else:
						# infer selection from transformed dimension
						X_train_transformed = selector.transform(X_train_encoded_float32)
						n_in  = X_train_encoded_float32.shape[1]
						n_out = X_train_transformed.shape[1]
						if n_out < n_in:
							support_mask = np.zeros(n_in, dtype=bool)
							support_mask[:n_out] = True
							# use transformed matrices for permutation importance
							X_train_perm_df = pd.DataFrame(
								X_train_transformed,
								columns=np.array(feature_encoded_lst_full)[support_mask].tolist()
							)
							X_test_transformed = selector.transform(X_test_encoded_float32)
							X_test_perm_df = pd.DataFrame(
								X_test_transformed,
								columns=np.array(feature_encoded_lst_full)[support_mask].tolist()
							)
							feature_encoded_perm_lst = np.array(feature_encoded_lst_full)[support_mask].tolist()
							message_perm_selection = (
								"The permutation importance was restricted to the features selected upstream (i.e., "
								+ str(len(feature_encoded_perm_lst))
								+ ") using an inferred selection mask because support_ was not available"
							)
						else:
							# no dimensionality reduction performed
							X_train_perm_df = X_train_encoded_float32.copy()
							X_test_perm_df  = X_test_encoded_float32.copy()
							feature_encoded_perm_lst = feature_encoded_lst_full.copy()
							message_perm_selection = (
								"The permutation importance was computed on all one-hot encoded features (i.e., "
								+ str(len(feature_encoded_perm_lst))
								+ ") because no dimensionality reduction was detected"
							)
				else:
					# no feature selection method was applied
					X_train_perm_df = X_train_encoded_float32.copy()
					X_test_perm_df  = X_test_encoded_float32.copy()
					feature_encoded_perm_lst = feature_encoded_lst_full.copy()
					message_perm_selection = (
						"The permutation importance was computed on all one-hot encoded features (i.e., "
						+ str(len(feature_encoded_perm_lst))
						+ ") because no feature selection method was used"
					)
			except Exception:
				# fallback if anything unexpected happens
				X_train_perm_df = X_train_encoded_float32.copy()
				X_test_perm_df  = X_test_encoded_float32.copy()
				feature_encoded_perm_lst = feature_encoded_lst_full.copy()
				message_perm_selection = (
					"The permutation importance defaulted to all one-hot encoded features (i.e., "
					+ str(len(feature_encoded_perm_lst))
					+ ") because selected features could not be recovered from the model"
				)
			# print a message
			print(message_perm_selection)
			# compute permutation importance
			try:
				# compute total number of permutations to estimate progress: one job per feature
				n_features = X_train_perm_df.shape[1]
				permutation_total = n_features
				if JOBS == 1:
					# when using a single thread, tqdm_joblib does not display intermediate updates
					# in this case, permutation_importance is executed normally without a progress bar
					permutation_train = permutation_importance(
						final_estimator, 
						X_train_perm_df, 
						y_train_series,
						n_repeats=NREPEATS, 
						random_state=42, # deterministic
						scoring='neg_log_loss', 
						n_jobs=1
					)
					permutation_test = permutation_importance(
						final_estimator, 
						X_test_perm_df, 
						y_test_series,
						n_repeats=NREPEATS, 
						random_state=42, # deterministic
						scoring='neg_log_loss', 
						n_jobs=1
					)
				else:
					# when using multiple threads, tqdm_joblib correctly displays the progress bar
					# permutation importance on training dataset
					with tqa.tqdm(
						total=permutation_total,
						desc="Permutation importance on the training dataset",
						position=0,
						leave=True,
						dynamic_ncols=True,
						mininterval=0.2
					) as progress_bar:
						with jl.parallel_backend('threading', n_jobs=JOBS):
							with tqjl.tqdm_joblib(progress_bar):
								with tpc.threadpool_limits(limits=1):
									with ctl.redirect_stdout(io.StringIO()), ctl.redirect_stderr(io.StringIO()):
										permutation_train = permutation_importance(
											final_estimator, 
											X_train_perm_df, 
											y_train_series,
											n_repeats=NREPEATS, 
											random_state=42, # deterministic
											scoring='neg_log_loss',
											n_jobs=JOBS
										)
					# permutation importance on testing dataset
					with tqa.tqdm(
						total=permutation_total,
						desc="Permutation importance on the testing dataset",
						position=0,
						leave=True,
						dynamic_ncols=True,
						mininterval=0.2
					) as progress_bar:
						with jl.parallel_backend('threading', n_jobs=JOBS):
							with tqjl.tqdm_joblib(progress_bar):
								with tpc.threadpool_limits(limits=1):
									with ctl.redirect_stdout(io.StringIO()), ctl.redirect_stderr(io.StringIO()):
										permutation_test = permutation_importance(
											final_estimator, 
											X_test_perm_df, 
											y_test_series,
											n_repeats=NREPEATS, 
											random_state=42,  # deterministic
											scoring='neg_log_loss',
											n_jobs=JOBS
										)
				# extract average permutation importance and its standard deviation (train)
				perm_train_mean = np.round(permutation_train.importances_mean, digits)
				perm_train_std  = np.round(permutation_train.importances_std,  digits)
				# extract average permutation importance and its standard deviation (test)
				perm_test_mean  = np.round(permutation_test.importances_mean,  digits)
				perm_test_std   = np.round(permutation_test.importances_std,   digits)
				# handle shape mismatch between names and scores
				min_len = min(
					len(feature_encoded_perm_lst),
					len(perm_train_mean),
					len(perm_test_mean)
				)
				feature_encoded_perm_lst = feature_encoded_perm_lst[:min_len]
				perm_train_mean = perm_train_mean[:min_len]
				perm_train_std  = perm_train_std[:min_len]
				perm_test_mean  = perm_test_mean[:min_len]
				perm_test_std   = perm_test_std[:min_len]
				# combine permutation importances from training and testing
				permutation_importance_df = pd.DataFrame({
					'feature': feature_encoded_perm_lst,
					'train_mean': perm_train_mean,
					'train_std':  perm_train_std,
					'test_mean':  perm_test_mean,
					'test_std':   perm_test_std
				}).sort_values(by='train_mean', ascending=False).reset_index(drop=True)
				# message to confirm success
				message_permutation = (
					"The permutation importance was successfully computed on both training and testing datasets"
				)
			except Exception as e:
				# fallback in case of failure: return empty DataFrame and report error
				permutation_importance_df = pd.DataFrame()
				message_permutation = (
					"An error occurred while computing permutation importance: " + str(e)
				)
		else:
			# if not requested, return empty DataFrame
			permutation_importance_df = pd.DataFrame()
			message_permutation = "The permutation importance was not computed"
		# print a message
		print(message_permutation)

		# perform prediction
		## from the training dataset
		y_pred_train = best_model.predict(X_train_encoded_float32)
		## from the testing dataset
		y_pred_test = best_model.predict(X_test_encoded_float32)

		# retrieve trained phenotype classes
		if CLASSIFIER == 'XGB':
			classes_lst = encoded_classes  # original string labels
			# build mapping: numeric class -> original string label
			class_mapping = dict(zip(range(len(encoded_classes)), encoded_classes))
			prob_columns = [class_mapping[c] for c in best_model.classes_]
		else:
			classes_lst = best_model.classes_
			prob_columns = best_model.classes_
		
		# retrieve p-values
		## as a numpy.ndarray from the training dataset
		y_pvalues_train_nda = best_model.predict_proba(X_train_encoded_float32)
		## as a numpy.ndarray from the testing dataset
		y_pvalues_test_nda = best_model.predict_proba(X_test_encoded_float32)
		## for binary XGB, make it 2-column if needed
		if y_pvalues_train_nda.ndim == 1 or y_pvalues_train_nda.shape[1] == 1:
			y_pvalues_train_nda = np.column_stack([1 - y_pvalues_train_nda, y_pvalues_train_nda])
			y_pvalues_test_nda  = np.column_stack([1 - y_pvalues_test_nda, y_pvalues_test_nda])
		## wrap into DataFrame with correct class labels
		y_pvalues_train_df = pd.DataFrame(y_pvalues_train_nda, columns=prob_columns)
		y_pvalues_test_df  = pd.DataFrame(y_pvalues_test_nda,  columns=prob_columns)
		## reindex to external class order (aligns properly)
		y_pvalues_train_df = y_pvalues_train_df.reindex(columns=classes_lst)
		y_pvalues_test_df  = y_pvalues_test_df.reindex(columns=classes_lst)
		# back to ndarray for compute_curve_metrics
		y_pvalues_train_nda = y_pvalues_train_df.to_numpy()
		y_pvalues_test_nda  = y_pvalues_test_df.to_numpy()

		# extract the confusion matrices (cm)
		## retrieve classes
		### transform numpy.ndarray into pandas.core.frame.DataFrame
		classes_df = pd.DataFrame(classes_lst)
		### rename variables of headers
		classes_df.rename(columns={0: 'phenotype'}, inplace=True)
		## extract confusion matrix from the training dataset
		### get the confusion matrix
		cm_classes_train_nda = confusion_matrix(y_train_series, y_pred_train)
		### transform numpy.ndarray into pandas.core.frame.DataFrame
		cm_classes_train_df = pd.DataFrame(cm_classes_train_nda, columns = classes_lst)
		### concatenate horizontally classes and confusion matrix
		cm_classes_train_df = pd.concat([classes_df, cm_classes_train_df], axis=1)
		## extract confusion matrix from the testing dataset
		### get the confusion matrix
		cm_classes_test_nda = confusion_matrix(y_test_series, y_pred_test)
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
		cm_metrics_train_nda = multilabel_confusion_matrix(y_train_series, y_pred_train)
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
		cm_metrics_test_nda = multilabel_confusion_matrix(y_test_series, y_pred_test)
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

		# prepare the true labels for computing ROC-AUC and PR-AUC:
		# for XGB, y_train_series and y_test_series are integer-encoded
		# we need to inverse-transform them back to their original string labels
		# so that they align with the probability columns in `classes_lst`
		# for other classifiers, the labels are already in the correct format
		if CLASSIFIER == 'XGB':
			y_train_series_for_metrics = pd.Series(
				class_encoder.inverse_transform(y_train_series), index=y_train_series.index
			)
			y_test_series_for_metrics = pd.Series(
				class_encoder.inverse_transform(y_test_series), index=y_test_series.index
			)
		else:
			y_train_series_for_metrics = y_train_series
			y_test_series_for_metrics = y_test_series

		# calculate the class-dependent metrics safely refactoring metrics calculations using np.where to avoid division by zero (i.e., result is safely set to 0 instead of nan or inf)
		# support = TP+FN
		# accuracy = (TP+TN)/(TP+FP+FN+TN)
		# sensitivity = TP/(TP+FN)
		# specificity = TN/(TN+FP)
		# precision = TP/(TP+FP)
		# recall = TP/(TP+FN)
		# f1-score = 2*(precision*recall)/(precision+recall)
		# MCC (Matthews Correlation Coefficient) = (TP*TN-FP*FN)/sqrt((TP+FP)*(TP+FN)*(TN+FP)*(TN+FN))
		# Cohen's kappa is not well-defined per class, as it is designed for overall pairwise agreement across the confusion matrix
		# curve-based metrics
		## ROC-AUC: area under the curve from receiver operating characteristic curve
		## PR-AUC: area under the curve from precision recall curve
		## PRG-AUC:  area under the curve from precision recall gain curve
		## from the training dataset
		### support
		metrics_classes_train_df['support'] = (metrics_classes_train_df.TP + metrics_classes_train_df.FN
		)
		### accuracy		
		train_acc_denom = metrics_classes_train_df.TP + metrics_classes_train_df.FP + metrics_classes_train_df.FN + metrics_classes_train_df.TN
		metrics_classes_train_df['accuracy'] = np.where(
			train_acc_denom == 0, 0,
			round((metrics_classes_train_df.TP + metrics_classes_train_df.TN) / train_acc_denom, digits)
		)
		### sensitivity		
		train_sens_denom = metrics_classes_train_df.TP + metrics_classes_train_df.FN
		metrics_classes_train_df['sensitivity'] = np.where(
			train_sens_denom == 0, 0,
			round(metrics_classes_train_df.TP / train_sens_denom, digits)
		)
		### specificity		
		train_spec_denom = metrics_classes_train_df.TN + metrics_classes_train_df.FP
		metrics_classes_train_df['specificity'] = np.where(
			train_spec_denom == 0, 0,
			round(metrics_classes_train_df.TN / train_spec_denom, digits)
		)
		### precision		
		train_prec_denom = metrics_classes_train_df.TP + metrics_classes_train_df.FP
		metrics_classes_train_df['precision'] = np.where(
			train_prec_denom == 0, 0,
			round(metrics_classes_train_df.TP / train_prec_denom, digits)
		)
		### recall		
		train_recall_denom = metrics_classes_train_df.TP + metrics_classes_train_df.FN
		metrics_classes_train_df['recall'] = np.where(
			train_recall_denom == 0, 0,
			round(metrics_classes_train_df.TP / train_recall_denom, digits)
		)
		### f1-score		
		train_f1_denom = metrics_classes_train_df.precision + metrics_classes_train_df.recall
		metrics_classes_train_df['f1-score'] = np.where(
			train_f1_denom == 0, 0,
			round(2 * metrics_classes_train_df.precision * metrics_classes_train_df.recall / train_f1_denom, digits)
		)
		### mcc
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
		### metrics curves
		metrics_curves_train_df = compute_curve_metrics(y_train_series_for_metrics, y_pvalues_train_nda, classes_lst, digits)
		metrics_classes_train_df = metrics_classes_train_df.join(metrics_curves_train_df.drop(columns=['phenotype']))
		
		## from the testing dataset
		### support
		metrics_classes_test_df['support'] = (metrics_classes_test_df.TP + metrics_classes_test_df.FN
		)
		### accuracy
		test_acc_denom = metrics_classes_test_df.TP + metrics_classes_test_df.FP + metrics_classes_test_df.FN + metrics_classes_test_df.TN
		metrics_classes_test_df['accuracy'] = np.where(
			test_acc_denom == 0, 0,
			round((metrics_classes_test_df.TP + metrics_classes_test_df.TN) / test_acc_denom, digits)
		)
		### sensitivity
		test_sens_denom = metrics_classes_test_df.TP + metrics_classes_test_df.FN
		metrics_classes_test_df['sensitivity'] = np.where(
			test_sens_denom == 0, 0,
			round(metrics_classes_test_df.TP / test_sens_denom, digits)
		)
		### specificity
		test_spec_denom = metrics_classes_test_df.TN + metrics_classes_test_df.FP
		metrics_classes_test_df['specificity'] = np.where(
			test_spec_denom == 0, 0,
			round(metrics_classes_test_df.TN / test_spec_denom, digits)
		)
		### precision
		test_prec_denom = metrics_classes_test_df.TP + metrics_classes_test_df.FP
		metrics_classes_test_df['precision'] = np.where(
			test_prec_denom == 0, 0,
			round(metrics_classes_test_df.TP / test_prec_denom, digits)
		)
		### recall
		test_recall_denom = metrics_classes_test_df.TP + metrics_classes_test_df.FN
		metrics_classes_test_df['recall'] = np.where(
			test_recall_denom == 0, 0,
			round(metrics_classes_test_df.TP / test_recall_denom, digits)
		)
		### f1-score
		test_f1_denom = metrics_classes_test_df.precision + metrics_classes_test_df.recall
		metrics_classes_test_df['f1-score'] = np.where(
			test_f1_denom == 0, 0,
			round(2 * metrics_classes_test_df.precision * metrics_classes_test_df.recall / test_f1_denom, digits)
		)
		### mcc
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
		### metrics curves
		metrics_curves_test_df  = compute_curve_metrics(y_test_series_for_metrics,  y_pvalues_test_nda,  classes_lst, digits)
		metrics_classes_test_df = metrics_classes_test_df.join(metrics_curves_test_df.drop(columns=['phenotype']))

		# QDA-specific validation: enforce finite predicted probabilities
		if CLASSIFIER == 'QDA':
			if np.isnan(y_pvalues_train_nda).any() or np.isnan(y_pvalues_test_nda).any():
				message_qda_probabilities = ("The QDA classifier produced NaN values in the predicted probabilities and the computation was aborted to prevent invalid metric calculations (Increase the 'reg_param' value, e.g., to 0.1 or 0.2, or reduce feature dimensionality)")
				raise ValueError(message_qda_probabilities)
			else:
				message_qda_probabilities = ("The QDA classifier did not produce NaN values in the predicted probabilities, and the computation continued normally")
				print(message_qda_probabilities)

		# ensure correct shape of y_pvalues for binary classification
		## from the training dataset
		if y_pvalues_train_nda.shape[1] == 2:
			y_score_train = y_pvalues_train_nda[:, 1] # positive class column
		else:
			y_score_train = y_pvalues_train_nda # already 1D or multiclass (ovr handled)
		## from the testing dataset
		if y_pvalues_test_nda.shape[1] == 2:
			y_score_test = y_pvalues_test_nda[:, 1] # positive class column
		else:
			y_score_test = y_pvalues_test_nda # already 1D or multiclass

		# fix for XGB binary classification — ensure correct column for positive class
		if CLASSIFIER == 'XGB' and len(classes_lst) == 2:
			# identify correct column index based on class_encoder mapping
			pos_index = list(class_encoder.classes_).index(classes_lst[1])
			y_score_train = y_pvalues_train_nda[:, pos_index]
			y_score_test  = y_pvalues_test_nda[:, pos_index]

		# flatten true labels (ensure 1D numeric labels for sklearn metrics)
		## from the training dataset
		y_train_nda = y_train_series.to_numpy().ravel()
		## from the testing dataset
		y_test_nda  = y_test_series.to_numpy().ravel()

		# determine positive label for binary classification
		if len(classes_lst) == 2:
			if CLASSIFIER == 'XGB':
				# for XGB, labels are encoded to integers (0/1), so use numeric
				positive_label = class_encoder.transform([classes_lst[1]])[0]
			else:
				positive_label = classes_lst[1] # string label for other classifiers
		else:
			positive_label = None

		# calculate the global metrics safely refactoring metrics calculations using np.nan_to_num to avoid division by zero (i.e., result is safely set to 0 instead of nan or inf)
		## from the training dataset
		### accuracy
		accuracy_train = round(np.nan_to_num(accuracy_score(y_train_series, y_pred_train)), digits)
		### sensitivity
		sensitivity_train = round(np.nan_to_num(imb.metrics.sensitivity_score(y_train_series, y_pred_train, average='macro')), digits)
		### specificity
		specificity_train = round(np.nan_to_num(imb.metrics.specificity_score(y_train_series, y_pred_train, average='macro')), digits)
		### precision
		precision_train = round(np.nan_to_num(precision_score(y_train_series, y_pred_train, average='macro', zero_division=0)), digits)
		### recall
		recall_train = round(np.nan_to_num(recall_score(y_train_series, y_pred_train, average='macro', zero_division=0)), digits)
		### f1 score
		f1_score_train = round(np.nan_to_num(f1_score(y_train_series, y_pred_train, average='macro', zero_division=0)), digits)
		### mcc
		mcc_train = round(np.nan_to_num(matthews_corrcoef(y_train_series, y_pred_train)), digits)
		### cohen kappa
		cohen_kappa_train = round(np.nan_to_num(cohen_kappa_score(y_train_series, y_pred_train)), digits)
		### ROC-AUC
		if len(classes_lst) > 2:
			# multiclass case → use one-vs-rest strategy for ROC-AUC
			roc_auc_train = round(roc_auc_score(y_train_nda, y_score_train, multi_class='ovr'), digits)
		else:
			# binary case → standard ROC-AUC (no multi_class argument needed)
			roc_auc_train = round(roc_auc_score(y_train_nda, y_score_train), digits)
		### PR-AUC
		if positive_label is not None:
			# binary case → specify the positive class explicitly (numeric or string depending on classifier)
			pr_auc_train = round(average_precision_score(y_train_nda, y_score_train, pos_label=positive_label), digits)
		else:
			# multiclass case → use macro-averaged PR-AUC across all classes
			pr_auc_train = round(average_precision_score(y_train_nda, y_score_train, average='macro'), digits)
		### PRG-AUC (macro)
		if len(classes_lst) == 2:
			# binary case → compute PRG-AUC directly using positive class probabilities
			y_binary_train = (y_train_nda == positive_label).astype(int)
			prg_auc_train, _, _, _ = compute_prg_auc(y_binary_train, y_score_train)
		else:
			# multiclass case → compute macro average of one-vs-rest PRG-AUC
			prg_auc_values = []  # store PRG-AUC per class
			for idx, cls in enumerate(classes_lst):
				# convert true labels into binary format for the current class
				y_binary = (y_train_nda == cls).astype(int)
				# extract predicted probabilities corresponding to the current class
				y_scores_cls = y_pvalues_train_nda[:, idx]
				# compute PRG-AUC for this class
				auc_cls, _, _, _ = compute_prg_auc(y_binary, y_scores_cls)
				prg_auc_values.append(auc_cls)
			# macro-average across classes (equal weight)
			prg_auc_train = np.mean(prg_auc_values)
		# round for readability
		prg_auc_train = round(prg_auc_train, digits)
		### PRG-AUC_clipped (macro)
		if len(classes_lst) == 2:
			# binary case → floor negative precision gains at 0
			y_binary_train = (y_train_nda == positive_label).astype(int)
			prg_auc_clipped_train, _, _, _ = compute_prg_auc(y_binary_train, y_score_train, clip_negative=True)
		else:
			# multiclass case → compute macro average of clipped PRG-AUC
			prg_auc_values = []  # store clipped PRG-AUC per class
			for idx, cls in enumerate(classes_lst):
				y_binary = (y_train_nda == cls).astype(int)
				y_scores_cls = y_pvalues_train_nda[:, idx]
				# compute PRG-AUC with clipping (no below-baseline penalties)
				auc_cls, _, _, _ = compute_prg_auc(y_binary, y_scores_cls, clip_negative=True)
				prg_auc_values.append(auc_cls)
			# macro-average across classes (equal weight)
			prg_auc_clipped_train = np.mean(prg_auc_values)
		# round for readability
		prg_auc_clipped_train = round(prg_auc_clipped_train, digits)
		## from the testing dataset
		### accuracy
		accuracy_test = round(np.nan_to_num(accuracy_score(y_test_series, y_pred_test)), digits)
		### sensitivity
		sensitivity_test = round(np.nan_to_num(imb.metrics.sensitivity_score(y_test_series, y_pred_test, average='macro')), digits)
		### specificity
		specificity_test = round(np.nan_to_num(imb.metrics.specificity_score(y_test_series, y_pred_test, average='macro')), digits)
		### precision
		precision_test = round(np.nan_to_num(precision_score(y_test_series, y_pred_test, average='macro', zero_division=0)), digits)
		### recall
		recall_test = round(np.nan_to_num(recall_score(y_test_series, y_pred_test, average='macro', zero_division=0)), digits)
		### f1 score
		f1_score_test = round(np.nan_to_num(f1_score(y_test_series, y_pred_test, average='macro', zero_division=0)), digits)
		### mcc
		mcc_test = round(np.nan_to_num(matthews_corrcoef(y_test_series, y_pred_test)), digits)
		### cohen kappa
		cohen_kappa_test = round(np.nan_to_num(cohen_kappa_score(y_test_series, y_pred_test)), digits)
		### ROC-AUC
		if len(classes_lst) > 2:
			# multiclass case → use one-vs-rest strategy for ROC-AUC
			roc_auc_test = round(roc_auc_score(y_test_nda, y_score_test, multi_class='ovr'), digits)
		else:
			# binary case → standard ROC-AUC (no multi_class argument needed)
			roc_auc_test = round(roc_auc_score(y_test_nda, y_score_test), digits)
		### PR-AUC
		if positive_label is not None:
			# binary case → specify the positive class explicitly (numeric or string depending on classifier)
			pr_auc_test = round(average_precision_score(y_test_nda, y_score_test, pos_label=positive_label), digits)
		else:
			# multiclass case → use macro-averaged PR-AUC across all classes
			pr_auc_test = round(average_precision_score(y_test_nda, y_score_test, average='macro'), digits)
		### PRG-AUC (macro)
		if len(classes_lst) == 2:
			# binary case → compute PRG-AUC directly using positive class probabilities
			y_binary_test = (y_test_nda == positive_label).astype(int)
			prg_auc_test, _, _, _ = compute_prg_auc(y_binary_test, y_score_test)
		else:
			# multiclass case → compute macro average of one-vs-rest PRG-AUC
			prg_auc_values = []  # store PRG-AUC per class
			for idx, cls in enumerate(classes_lst):
				y_binary = (y_test_nda == cls).astype(int)
				y_scores_cls = y_pvalues_test_nda[:, idx]
				# compute PRG-AUC for this class
				auc_cls, _, _, _ = compute_prg_auc(y_binary, y_scores_cls)
				prg_auc_values.append(auc_cls)
			# macro-average across classes (equal weight)
			prg_auc_test = np.mean(prg_auc_values)
		# round for readability
		prg_auc_test = round(prg_auc_test, digits)
		### PRG-AUC_clipped (macro)
		if len(classes_lst) == 2:
			# binary case → floor negative precision gains at 0
			y_binary_test = (y_test_nda == positive_label).astype(int)
			prg_auc_clipped_test, _, _, _ = compute_prg_auc(y_binary_test, y_score_test, clip_negative=True)
		else:
			# multiclass case → compute macro average of clipped PRG-AUC
			prg_auc_values = []  # store clipped PRG-AUC per class
			for idx, cls in enumerate(classes_lst):
				y_binary = (y_test_nda == cls).astype(int)
				y_scores_cls = y_pvalues_test_nda[:, idx]
				# compute PRG-AUC with clipping (no below-baseline penalties)
				auc_cls, _, _, _ = compute_prg_auc(y_binary, y_scores_cls, clip_negative=True)
				prg_auc_values.append(auc_cls)
			# macro-average across classes (equal weight)
			prg_auc_clipped_test = np.mean(prg_auc_values)
		# round for readability
		prg_auc_clipped_test = round(prg_auc_clipped_test, digits)
		## combine in dataframes
		metrics_global_train_df = pd.DataFrame({
			'accuracy': [round(accuracy_train, digits)], 
			'sensitivity': [round(sensitivity_train, digits)], 
			'specificity': [round(specificity_train, digits)], 
			'precision': [round(precision_train, digits)], 
			'recall': [round(recall_train, digits)], 
			'f1-score': [round(f1_score_train, digits)], 
			'MCC': [round(mcc_train, digits)], 
			'Cohen kappa': [round(cohen_kappa_train, digits)], 
			'ROC-AUC': [round(roc_auc_train, digits)], 
			'PR-AUC': [round(pr_auc_train, digits)], 
			'PRG-AUC': [round(prg_auc_train, digits)],
			'PRG-AUC_clipped': [round(prg_auc_clipped_train, digits)]
			})
		metrics_global_test_df = pd.DataFrame({
			'accuracy': [round(accuracy_test, digits)], 
			'sensitivity': [round(sensitivity_test, digits)], 
			'specificity': [round(specificity_test, digits)], 
			'precision': [round(precision_test, digits)], 
			'recall': [round(recall_test, digits)], 
			'f1-score': [round(f1_score_test, digits)], 
			'MCC': [round(mcc_test, digits)], 
			'Cohen kappa': [round(cohen_kappa_test, digits)], 
			'ROC-AUC': [round(roc_auc_test, digits)], 
			'PR-AUC': [round(pr_auc_test, digits)], 
			'PRG-AUC': [round(prg_auc_test, digits)],
			'PRG-AUC_clipped': [round(prg_auc_clipped_test, digits)]
			})

		# combine expectation and prediction from the training dataset
		## convert numpy.ndarray into a dataframe with explicit column name
		y_pred_train_df = pd.DataFrame(y_pred_train, columns=["prediction"])
		## rebuild a clean dataframe for expectations using preserved sample IDs
		y_train_df = pd.DataFrame({
			"sample": train_sample_ids, # explicit sample identifiers
			"expectation": y_train_series.values # always 1-dimensional
		})
		## concatenate horizontally with index resets
		combined_train_df = pd.concat(
			[y_train_df.reset_index(drop=True), # avoids index misalignment
			y_pred_train_df.reset_index(drop=True)], # avoids index misalignment
			axis=1,
			join="inner" # safeguards against accidental row misalignment
		)

		# combine expectation and prediction from the testing dataset
		## convert numpy.ndarray into a dataframe with explicit column name
		y_pred_test_df = pd.DataFrame(y_pred_test, columns=["prediction"])
		## rebuild a clean dataframe for expectations using preserved sample IDs
		y_test_df = pd.DataFrame({
			"sample": test_sample_ids, # explicit sample identifiers
			"expectation": y_test_series.values # always 1-dimensional
		})
		## concatenate horizontally with index resets
		combined_test_df = pd.concat(
			[y_test_df.reset_index(drop=True), # avoids index misalignment
			y_pred_test_df.reset_index(drop=True)], # avoids index misalignment
			axis=1,
			join="inner" # safeguards against accidental row misalignment
		)
		## transform back phenotype numbers into phenotype classes for the XGB classifier
		## (XGB is the only classifier that receives encoded class labels)
		if CLASSIFIER == 'XGB':
			combined_train_df["expectation"] = class_encoder.inverse_transform(combined_train_df["expectation"])
			combined_train_df["prediction"]  = class_encoder.inverse_transform(combined_train_df["prediction"])
			combined_test_df["expectation"]  = class_encoder.inverse_transform(combined_test_df["expectation"])
			combined_test_df["prediction"]   = class_encoder.inverse_transform(combined_test_df["prediction"])

		# concatenate horizontally the combined predictions and p-values
		## from the training dataset
		combined_train_df = pd.concat(
			[combined_train_df.reset_index(drop=True), # avoids index misalignment during concatenation
			y_pvalues_train_df.reset_index(drop=True)], # avoids index misalignment during concatenation
			axis=1, join="inner" # safeguards against accidental row misalignment down the line
		)
		## from the testing dataset
		combined_test_df = pd.concat(
			[combined_test_df.reset_index(drop=True), # avoids index misalignment during concatenation
			y_pvalues_test_df.reset_index(drop=True)], # avoids index misalignment during concatenation
			axis=1, join="inner" # safeguards against accidental row misalignment down the line
		)

		# round digits of all numeric columns (use .copy() to avoid SettingWithCopyWarning)
		## from the training dataset
		combined_train_df = combined_train_df.copy()
		numeric_cols = combined_train_df.select_dtypes(include="number").columns
		combined_train_df[numeric_cols] = (
			combined_train_df[numeric_cols].astype(float).round(digits)
		)
		## from the testing dataset
		combined_test_df = combined_test_df.copy()
		numeric_cols = combined_test_df.select_dtypes(include="number").columns
		combined_test_df[numeric_cols] = (
			combined_test_df[numeric_cols].astype(float).round(digits)
		)

		# ensure samples are sorted in alphanumerical order
		## from the training dataset
		combined_train_df = (
			combined_train_df
			.sort_values(by="sample", key=lambda col: col.astype(str))
			.reset_index(drop=True)
		)
		## from the testing dataset
		combined_test_df = (
			combined_test_df
			.sort_values(by="sample", key=lambda col: col.astype(str))
			.reset_index(drop=True)
		)

		# build a clean phenotype/dataset file to reuse later
		## keep only sample identifiers and the phenotype column (the second column in the combined DF)
		simplified_train_df = combined_train_df.iloc[:, 0:2].copy()
		simplified_test_df  = combined_test_df.iloc[:, 0:2].copy()
		## annotate dataset origin
		simplified_train_df["dataset"] = "training"
		simplified_test_df["dataset"]  = "testing"
		## concatenate vertically while safeguarding against accidental column mismatch
		simplified_train_test_df = pd.concat(
			[simplified_train_df, simplified_test_df],
			ignore_index=True,
			axis=0,
			join="inner"
		)
		## rename phenotype column for clarity (the second column)
		simplified_train_test_df = simplified_train_test_df.rename(
			columns={simplified_train_test_df.columns[1]: "phenotype"}
		).copy()
		## ensure sample identifiers are sorted and index is reset
		simplified_train_test_df = (
			simplified_train_test_df
			.sort_values("sample")
			.reset_index(drop=True)
		)

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
		outpath_feature_encoder = OUTPUTPATH + '/' + PREFIX + '_feature_encoder' + '.obj'
		outpath_class_encoder = OUTPUTPATH + '/' + PREFIX + '_class_encoder' + '.obj'
		outpath_model = OUTPUTPATH + '/' + PREFIX + '_model' + '.obj'
		outpath_scores_parameters = OUTPUTPATH + '/' + PREFIX + '_scores_parameters' + '.tsv'
		outpath_feature_importance = OUTPUTPATH + '/' + PREFIX + '_feature_importances' + '.tsv'
		if PERMUTATIONIMPORTANCE is True:
			outpath_permutation_importance = OUTPUTPATH + '/' + PREFIX + '_permutation_importances' + '.tsv'
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
		all_scores_parameters_df.to_csv(outpath_scores_parameters, sep="\t", index=False, header=True)
		feature_importance_df.to_csv(outpath_feature_importance, sep="\t", index=False, header=True)
		if PERMUTATIONIMPORTANCE is True:
			permutation_importance_df.to_csv(outpath_permutation_importance, sep="\t", index=False, header=True)
		count_classes_df.to_csv(outpath_count_classes, sep="\t", index=False, header=True)		
		cm_classes_train_df.to_csv(outpath_cm_classes_train, sep="\t", index=False, header=True)
		cm_classes_test_df.to_csv(outpath_cm_classes_test, sep="\t", index=False, header=True)
		metrics_classes_train_df.to_csv(outpath_metrics_classes_train, sep="\t", index=False, header=True)
		metrics_classes_test_df.to_csv(outpath_metrics_classes_test, sep="\t", index=False, header=True)
		metrics_global_train_df.to_csv(outpath_metrics_global_train, sep="\t", index=False, header=True)
		metrics_global_test_df.to_csv(outpath_metrics_global_test, sep="\t", index=False, header=True)
		combined_train_df.to_csv(outpath_train, sep="\t", index=False, header=True)
		combined_test_df.to_csv(outpath_test, sep="\t", index=False, header=True)
		simplified_train_test_df.to_csv(outpath_phenotype_dataset, sep="\t", index=False, header=True)
		## save the training features
		with open(outpath_features, 'wb') as file:
			pi.dump(features, file)
		## save the feature encoder
		with open(outpath_feature_encoder, 'wb') as file:
			pi.dump(feature_encoder, file)
		## save the class encoder for the XGB model
		if CLASSIFIER == 'XGB':
			with open(outpath_class_encoder, 'wb') as file:
				pi.dump(class_encoder, file)
		## save the model
		with open(outpath_model, 'wb') as file:
			pi.dump(best_model, file)
		## write output in a txt file
		log_file = open(outpath_log, "w")
		log_file.writelines(["###########################\n######### context #########\n###########################\n"])
		print(context, file=log_file)
		log_file.writelines(["###########################\n######## reference ########\n###########################\n"])
		print(reference, file=log_file)
		log_file.writelines(["###########################\n###### repositories  ######\n###########################\n"])
		print(parser.epilog, file=log_file)
		log_file.writelines(["###########################\n#### acknowledgements  ####\n###########################\n"])
		print(acknowledgements, file=log_file)
		log_file.writelines(["###########################\n######## versions  ########\n###########################\n"])
		log_file.writelines("GenomicBasedClassification: " + __version__ + " (released in " + __release__ + ")" + "\n")
		log_file.writelines("python: " + str(sys.version_info[0]) + "." + str(sys.version_info[1]) + "\n")
		log_file.writelines("argparse: " + str(ap.__version__) + "\n")
		log_file.writelines("imblearn: " + str(imb.__version__) + "\n")
		log_file.writelines("numpy: " + str(np.__version__) + "\n")
		log_file.writelines("pandas: " + str(pd.__version__) + "\n")
		log_file.writelines("pickle: " + str(pi.format_version) + "\n")
		log_file.writelines("joblib: " + str(jl.__version__) + "\n")
		log_file.writelines("tqdm: " + str(tq.__version__) + "\n")
		log_file.writelines("tqdm-joblib: " + str(imp.version("tqdm-joblib")) + "\n")
		log_file.writelines("sklearn: " + str(sk.__version__) + "\n")
		log_file.writelines("catboost: " + str(imp.version("catboost")) + "\n")
		log_file.writelines("lightgbm: " + str(lgbm.__version__) + "\n")
		log_file.writelines("xgboost: " + str(xgb.__version__) + "\n")
		log_file.writelines(["###########################\n######## arguments  #######\n###########################\n"])
		for key, value in vars(args).items():
			log_file.write(f"{key}: {value}\n")
		log_file.writelines(["###########################\n######### samples  ########\n###########################\n"])
		print(count_classes_df.to_string(index=False), file=log_file)
		log_file.writelines(["###########################\n######### checks  #########\n###########################\n"])
		log_file.writelines(message_traceback + "\n")
		log_file.writelines(message_warnings + "\n")
		log_file.writelines(message_versions + "\n")
		log_file.writelines(message_subcommand + "\n")
		log_file.writelines(message_limit + "\n")
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
		log_file.writelines(message_missing_features + "\n")
		log_file.writelines(message_extra_features + "\n")
		log_file.writelines(message_assert_encoded_features + "\n")
		log_file.writelines(message_column_order + "\n")
		log_file.writelines(message_ohe_features + "\n")
		log_file.writelines(message_feature_selection + "\n")
		log_file.writelines(message_classifier + "\n")
		if CLASSIFIER == 'CAT':
			log_file.writelines(message_CAT_type_phenotype_classes + "\n")
		if CLASSIFIER == 'XGB':
			log_file.writelines(message_XGB_type_phenotype_classes + "\n")
		log_file.writelines(message_pipeline + "\n")
		log_file.writelines(message_parameters + "\n")
		log_file.writelines(message_metrics_cv + "\n")
		log_file.writelines(message_parallelization + "\n")
		log_file.writelines(message_best_parameters + "\n")
		log_file.writelines(message_best_score + "\n")


		log_file.writelines(message_selected_features + "\n")
		log_file.writelines(message_importance_encoded_feature_names + "\n")
		log_file.writelines(message_importance_count + "\n")
		log_file.writelines(message_compatibility_permutation_nrepeat + "\n")
		if PERMUTATIONIMPORTANCE is True:
			log_file.writelines(message_perm_selection + "\n")
		log_file.writelines(message_permutation + "\n")
		if CLASSIFIER == 'QDA':
			log_file.writelines(message_qda_probabilities + "\n")
		log_file.writelines(message_output_directory + "\n")
		log_file.writelines(["###########################\n####### execution  ########\n###########################\n"])
		log_file.writelines("The script started on " + str(step1_start) + "\n")
		log_file.writelines("The script stoped on " + str(step1_end) + "\n")
		total_secs = step1_diff.total_seconds() # store total seconds before modification
		secs = total_secs # use a working copy for breakdown
		days,secs = divmod(secs,secs_per_day:=60*60*24)
		hrs,secs = divmod(secs,secs_per_hr:=60*60)
		mins,secs = divmod(secs,secs_per_min:=60)
		secs = round(secs, 2)
		message_duration = 'The script lasted {} days, {} hrs, {} mins and {} secs (i.e., {} secs in total)'.format(
			int(days), int(hrs), int(mins), secs, round(total_secs, 2)
		)
		log_file.writelines(message_duration + "\n")
		log_file.writelines(["###########################\n###### output  files ######\n###########################\n"])
		log_file.writelines(outpath_count_classes + "\n")
		log_file.writelines(outpath_features + "\n")
		log_file.writelines(outpath_feature_encoder + "\n")
		if CLASSIFIER == 'XGB':
			log_file.writelines(outpath_class_encoder + "\n")
		log_file.writelines(outpath_model + "\n")
		log_file.writelines(outpath_scores_parameters + "\n")
		log_file.writelines(outpath_feature_importance + "\n")
		if PERMUTATIONIMPORTANCE is True:
			log_file.writelines(outpath_permutation_importance + "\n")
		log_file.writelines(outpath_cm_classes_train + "\n")
		log_file.writelines(outpath_cm_classes_test + "\n")
		log_file.writelines(outpath_metrics_classes_train + "\n")
		log_file.writelines(outpath_metrics_classes_test + "\n")
		log_file.writelines(outpath_metrics_global_train + "\n")
		log_file.writelines(outpath_metrics_global_test + "\n")
		log_file.writelines(outpath_train + "\n")
		log_file.writelines(outpath_test + "\n")
		log_file.writelines(outpath_phenotype_dataset + "\n")
		log_file.writelines(outpath_log + "\n")
		log_file.writelines(["###########################\n### feature  importance ###\n###########################\n"])
		print(feature_importance_df.head(20).to_string(index=False), file=log_file)
		log_file.writelines(f"Note: Up to 20 results are displayed in the log for monitoring purposes, while the full set of results is available in the output files. \n")
		log_file.writelines(f"Note: NaN placeholder in case no native or detectable feature importance is available. \n")
		log_file.writelines(f"Note: AdaBoost (ADA) typically produces binary feature importances (0 or 1) because each weak learner splits on only one feature. \n")
		log_file.writelines(f"Note: Boosting models, especially Histogram-based Gradient Boosting (HGB), may yield all-zero feature importances when no meaningful split gains are computed—typically due to strong regularization, shallow trees, or low feature variability. \n")
		log_file.writelines(["###########################\n# permutation  importance #\n###########################\n"])
		if PERMUTATIONIMPORTANCE is True:
			print(permutation_importance_df.head(20).to_string(index=False), file=log_file)
			log_file.writelines(f"Note: Up to 20 results are displayed in the log for monitoring purposes, while the full set of results is available in the output files. \n")
			log_file.writelines(f"Note: Positive permutation importance values indicate features that contribute positively to the model’s performance, while negative values suggest features that degrade performance when included. \n")
		else:
			log_file.writelines(f"Note: Permutation importance was not requested. \n")
		log_file.writelines(["###########################\n#### confusion  matrix ####\n###########################\n"])
		log_file.writelines(f"from the training dataset: \n")
		print(cm_classes_train_df.to_string(index=False), file=log_file)
		log_file.writelines(f"from the testing dataset: \n")
		print(cm_classes_test_df.to_string(index=False), file=log_file)
		log_file.writelines(f"Note: The expectation and prediction are represented by rows and columns, respectively. \n")
		log_file.writelines(["###########################\n#### metrics  per class ###\n###########################\n"])
		log_file.writelines(f"from the training dataset: \n")
		print(metrics_classes_train_df.to_string(index=False), file=log_file)
		log_file.writelines(f"from the testing dataset: \n")
		print(metrics_classes_test_df.to_string(index=False), file=log_file)
		log_file.writelines(f"Note: The term 'support' corresponds to TP + FN. \n")
		log_file.writelines(f"Note: MCC stands for Matthews Correlation Coefficient. \n")
		log_file.writelines(f"Note: Sensitivity and recall must be equal, as they are based on the same formula. \n")
		log_file.writelines(f"Note: In the case of binary phenotype, the negative and positive classes correspond to the first and second classes in alphabetical order, respectively. \n")
		log_file.writelines(f"Note: PRG-AUC can be negative, reflecting below-baseline performance expected with imbalanced data, while PRG-AUC_clipped floors negative gains to zero to separate total performance (including penalties) from above-baseline contributions, coinciding with PRG-AUC when the model stays above baseline. \n")
		log_file.writelines(f"Note: A PRG-AUC close to 0.5 for a specific class does not necessarily indicate poor performance. In binary or one-vs-rest evaluations, it reflects limited precision gain over the random baseline (e.g., SVC). \n")
		log_file.writelines(["###########################\n##### global  metrics #####\n###########################\n"])
		log_file.writelines(f"from the training dataset: \n")
		print(metrics_global_train_df.to_string(index=False), file=log_file)
		log_file.writelines(f"from the testing dataset: \n")
		print(metrics_global_test_df.to_string(index=False), file=log_file)
		log_file.writelines(f"Note: MCC stands for Matthews Correlation Coefficient. \n")
		log_file.writelines(f"Note: Sensitivity and recall must be equal, as they are based on the same formula. \n")
		log_file.writelines(f"Note: PRG-AUC can be negative, reflecting below-baseline performance expected with imbalanced data, while PRG-AUC_clipped floors negative gains to zero to separate total performance (including penalties) from above-baseline contributions, coinciding with PRG-AUC when the model stays above baseline. \n")
		log_file.writelines(f"Note: A global PRG-AUC close to 0.5 does not necessarily indicate poor model performance. In binary problems, it reflects limited precision gain over the random baseline (e.g., SVC). In multiclass settings, it can also result from averaging one-vs-rest curves (e.g., XGB), even when per-class PRG-AUC values are higher. \n")
		log_file.writelines(["###########################\n#### training  dataset ####\n###########################\n"])
		print(combined_train_df.head(20).to_string(index=False), file=log_file)
		log_file.writelines(f"Note: Up to 20 results are displayed in the log for monitoring purposes, while the full set of results is available in the output files. \n")
		log_file.writelines(f"Note: The p-values are reported for each phenotype of interest. \n")
		log_file.writelines(["###########################\n##### testing dataset #####\n###########################\n"])
		print(combined_test_df.head(20).to_string(index=False), file=log_file)
		log_file.writelines(f"Note: Up to 20 results are displayed in the log for monitoring purposes, while the full set of results is available in the output files. \n")
		log_file.writelines(f"Note: The p-values are reported for each phenotype of interest. \n")
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
		#### check if at least one sample and 3 columns
		if (rows_mutations >= 1) and (columns_mutations >= 3): 
			message_input_mutations = "The minimum required number of samples in the dataset (i.e., >= 1) and the expected number of columns (i.e., >= 3) in the input file of mutations were properly controlled (i.e., " + str(rows_mutations) + " and " + str(columns_mutations) + " , respectively)"
			print(message_input_mutations)
		else: 
			message_input_mutations = "The minimum required number of samples in the dataset (i.e., 1) and the expected number of columns (i.e., >= 3) in the input file of mutations were not properly controlled (i.e., " + str(rows_mutations) + " and " + str(columns_mutations) + " , respectively)"
			raise Exception(message_input_mutations)
		## training features
		with open(INPUTPATH_FEATURES, 'rb') as file:
			features = pi.load(file)
		## feature encoder
		with open(INPUTPATH_FEATURE_ENCODER, 'rb') as file:
			feature_encoder = pi.load(file)
		## encoded classes for the XGB model
		if INPUTPATH_CLASS_ENCODER == None:
			message_class_encoder = "The class encoder were not provided"
			print(message_class_encoder)		
		else:
			message_class_encoder = "The class encoder were provided"
			print(message_class_encoder)
			with open(INPUTPATH_CLASS_ENCODER, 'rb') as file:
				class_encoder = pi.load(file)
		## model
		with open(INPUTPATH_MODEL, 'rb') as file:
			loaded_model = pi.load(file)

		# prepare data
		## replace missing genomic data by a string
		df_mutations = df_mutations.fillna('missing')
		## rename labels of headers
		df_mutations.rename(columns={df_mutations.columns[0]: 'sample'}, inplace=True)
		## sort by samples
		df_mutations = df_mutations.sort_values(by='sample')
		## prepare mutations indexing the sample columns
		X_mutations = df_mutations.set_index('sample')

		# encode categorical data into binary data using the one-hot encoder from the modeling subcommand
		## check missing features
		missing_features = set(features) - set(X_mutations.columns)
		if missing_features:
			message_missing_features = "The following training features expected by the one-hot encoder are missing in the input tested mutations: " + str(sorted(missing_features))
			raise Exception(message_missing_features)
		else: 
			message_missing_features = "The input tested mutations include all features required by the trained one-hot encoder"
			print (message_missing_features)
		## check extra features
		extra_features = set(X_mutations.columns) - set(features)
		if extra_features:
			message_extra_features = "The following unexpected features in the input tested mutations will be ignored for one-hot encoding: " + str(sorted(extra_features))
			print (message_extra_features)
		else: 
			message_extra_features = "The input tested mutations contain no unexpected features for one hot encoding"
			print (message_extra_features)
		## ensure feature column order and cast to str
		X_features_str = X_mutations[features].astype(str)
		## apply the trained OneHotEncoder to the selected input features
		X_mutations_encoded = feature_encoder.transform(X_features_str)
		## assert identical encoded features between training and prediction datasets
		training_encoded_features = list(feature_encoder.get_feature_names_out())
		prediction_encoded_features = list(X_mutations_encoded.columns)
		if training_encoded_features != prediction_encoded_features:
			message_assert_encoded_features = "The encoded features between training and prediction datasets do not match"
			raise AssertionError(message_assert_encoded_features)
		else:
			message_assert_encoded_features = "The encoded features between training and prediction datasets were confirmed as identical"
			print(message_assert_encoded_features)

		# count features for diagnostics
		## count the number of raw categorical features before one-hot encoding
		features_before_ohe_int = len(features)
		## count the number of binary features after one-hot encoding
		features_after_ohe_int = X_mutations_encoded.shape[1]
		## print a message
		message_ohe_features = "The " + str(features_before_ohe_int) + " provided features were one-hot encoded into " + str(features_after_ohe_int) + " encoded features"
		print(message_ohe_features)
		## count the number of features used by the model
		selected_features_int = count_selected_features(loaded_model, X_mutations_encoded)
		## print a message
		message_selected_features = "The pipeline expected " + str(selected_features_int) + " one-hot encoded features to perform prediction"
		print(message_selected_features)

		# count and print phenotype classes
		## retrieve trained phenotype classes
		if INPUTPATH_CLASS_ENCODER != None:
			encoded_classes = class_encoder.classes_
			classes_lst = encoded_classes
		else:
			classes_lst = loaded_model.classes_
		## extract phenotype class as a string
		classes_str = classes_str = ", ".join(f"'{item}'" for item in classes_lst)
		## count phenotype class
		counts_classes_int = len(classes_lst)
		## print a message
		message_number_phenotype_classes = "The provided best model harbored " + str(counts_classes_int) + " classes: " + classes_str
		print(message_number_phenotype_classes)

		# detect classifier type even if wrapped in a Pipeline
		if hasattr(loaded_model, 'named_steps') and 'model' in loaded_model.named_steps:
			final_estimator = loaded_model.named_steps['model']
		else:
			final_estimator = loaded_model
		detected_model = final_estimator.__class__.__name__

		# print a message about pipeline components
		message_detected_model = (
			"The pipeline components of the provided best model were properly recognized: " + re.sub(r'\s+', ' ', str(loaded_model)).strip())
		print(message_detected_model)

		# determine expected feature order from the trained pipeline
		if hasattr(loaded_model, "feature_names_in_"):
			# The pipeline itself remembers training feature names (since sklearn 1.0+)
			expected_features = list(loaded_model.feature_names_in_)
		elif hasattr(loaded_model, "named_steps") and "feature_selection" in loaded_model.named_steps:
			# Use the input feature names stored in the selector (without validation)
			expected_features = getattr(
				loaded_model.named_steps["feature_selection"], "feature_names_in_", X_mutations_encoded.columns
			)
		else:
			# Fallback to whatever is available
			expected_features = X_mutations_encoded.columns
		# Align prediction matrix to expected feature names and order
		X_mutations_encoded = X_mutations_encoded.reindex(columns=expected_features, fill_value=0)
		message_prediction_alignment = ("The one-hot encoded prediction matrix was reindexed and aligned to match the exact feature names and order expected by the trained pipeline")
		print(message_prediction_alignment)

		# perform prediction
		y_pred_mutations = loaded_model.predict(X_mutations_encoded)

		# check compatibility between model and class encoder required for XGB model
		if (INPUTPATH_CLASS_ENCODER != None) and (detected_model != "XGBClassifier"):
			message_compatibility_model_classes = ("The classifier of the provided best model was not XGB and did not require to provide class encoder")
			raise Exception(message_compatibility_model_classes)
		elif (INPUTPATH_CLASS_ENCODER == None) and (detected_model == "XGBClassifier"):
			message_compatibility_model_classes = ("The classifier of the provided best model was XGB and required to provide class encoder")
			raise Exception(message_compatibility_model_classes)
		else:
			message_compatibility_model_classes = ("The classifier of the provided best model was verified for compatibility with class encoder, which are only used for the XGB classifier")
			print(message_compatibility_model_classes)

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
		if INPUTPATH_CLASS_ENCODER != None:
			combined_mutations_df["prediction"] = class_encoder.inverse_transform(combined_mutations_df["prediction"])

		# retrieve p-values
		## as a numpy.ndarray
		y_pvalues_mutations_nda = loaded_model.predict_proba(X_mutations_encoded)
		## transform numpy.ndarray into pandas.core.frame.DataFrame
		y_pvalues_mutations_df = pd.DataFrame(y_pvalues_mutations_nda, columns = classes_lst)

		# concatenate horizontally the combined predictions and p-values without reset index
		combined_mutations_df = pd.concat([combined_mutations_df, y_pvalues_mutations_df], axis=1)

		# round digits of all numeric columns (use .copy() to avoid SettingWithCopyWarning)
		combined_mutations_df = combined_mutations_df.copy()
		numeric_cols = combined_mutations_df.select_dtypes(include="number").columns
		combined_mutations_df[numeric_cols] = combined_mutations_df[numeric_cols].astype(float).round(digits)

		# ensure samples are sorted in alphanumerical order
		combined_mutations_df = (
			combined_mutations_df
			.sort_values(by="sample", key=lambda col: col.astype(str))
			.reset_index(drop=True)
		)

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
		log_file.writelines(["###########################\n######### context #########\n###########################\n"])
		print(context, file=log_file)
		log_file.writelines(["###########################\n######## reference ########\n###########################\n"])
		print(reference, file=log_file)
		log_file.writelines(["###########################\n###### repositories  ######\n###########################\n"])
		print(parser.epilog, file=log_file)
		log_file.writelines(["###########################\n#### acknowledgements  ####\n###########################\n"])
		print(acknowledgements, file=log_file)
		log_file.writelines(["###########################\n######## versions  ########\n###########################\n"])
		log_file.writelines("GenomicBasedClassification: " + __version__ + " (released in " + __release__ + ")" + "\n")
		log_file.writelines("python: " + str(sys.version_info[0]) + "." + str(sys.version_info[1]) + "\n")
		log_file.writelines("argparse: " + str(ap.__version__) + "\n")
		log_file.writelines("imblearn: " + str(imb.__version__) + "\n")
		log_file.writelines("pandas: " + str(pd.__version__) + "\n")
		log_file.writelines("sklearn: " + str(sk.__version__) + "\n")
		log_file.writelines("pickle: " + str(pi.format_version) + "\n")
		log_file.writelines("catboost: " + str(imp.version("catboost")) + "\n")
		log_file.writelines("lightgbm: " + str(lgbm.__version__) + "\n")
		log_file.writelines("xgboost: " + str(xgb.__version__) + "\n")
		log_file.writelines("numpy: " + str(np.__version__) + "\n")
		log_file.writelines("joblib: " + str(jl.__version__) + "\n")
		log_file.writelines("tqdm: " + str(tq.__version__) + "\n")
		log_file.writelines("tqdm-joblib: " + str(imp.version("tqdm-joblib")) + "\n")
		log_file.writelines(["###########################\n######## arguments  #######\n###########################\n"])
		for key, value in vars(args).items():
			log_file.write(f"{key}: {value}\n")
		log_file.writelines(["###########################\n######### checks  #########\n###########################\n"])
		log_file.writelines(message_traceback + "\n")
		log_file.writelines(message_warnings + "\n")
		log_file.writelines(message_versions + "\n")
		log_file.writelines(message_subcommand + "\n")
		log_file.writelines(message_input_mutations + "\n")		
		log_file.writelines(message_class_encoder + "\n")
		log_file.writelines(message_missing_features + "\n")
		log_file.writelines(message_extra_features + "\n")
		log_file.writelines(message_assert_encoded_features + "\n")
		log_file.writelines(message_ohe_features + "\n")
		log_file.writelines(message_selected_features + "\n")
		log_file.writelines(message_number_phenotype_classes + "\n")
		log_file.writelines(message_detected_model + "\n")
		log_file.writelines(message_prediction_alignment + "\n")
		log_file.writelines(message_compatibility_model_classes + "\n")
		log_file.writelines(message_output_directory + "\n")
		log_file.writelines(["###########################\n####### execution  ########\n###########################\n"])
		log_file.writelines("The script started on " + str(step1_start) + "\n")
		log_file.writelines("The script stoped on " + str(step1_end) + "\n")
		total_secs = step1_diff.total_seconds() # store total seconds before modification
		secs = total_secs # use a working copy for breakdown
		days,secs = divmod(secs,secs_per_day:=60*60*24)
		hrs,secs = divmod(secs,secs_per_hr:=60*60)
		mins,secs = divmod(secs,secs_per_min:=60)
		secs = round(secs, 2)
		message_duration = 'The script lasted {} days, {} hrs, {} mins and {} secs (i.e., {} secs in total)'.format(
			int(days), int(hrs), int(mins), secs, round(total_secs, 2)
		)
		log_file.writelines(message_duration + "\n")
		log_file.writelines(["###########################\n###### output  files ######\n###########################\n"])
		log_file.writelines(outpath_prediction + "\n")
		log_file.writelines(outpath_log + "\n")
		log_file.writelines(["###########################\n### prediction  dataset ###\n###########################\n"])
		print(combined_mutations_df.head(20).to_string(index=False), file=log_file)
		log_file.writelines(f"Note: Up to 20 results are displayed in the log for monitoring purposes, while the full set of results is available in the output files. \n")
		log_file.writelines(f"Note: The p-values are reported for each phenotype of interest. \n")
		log_file.close()
	# print final messages
	print(message_duration)
	print("The results are ready: " + OUTPUTPATH)
	print(parser.epilog)

# identify the block which will only be run when the script is executed directly
if __name__ == "__main__":
	main()
