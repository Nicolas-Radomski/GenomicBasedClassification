# Usage
The repository GenomicBasedClassification provides a Python (recommended version 3.12) script called GenomicBasedClassification.py to perform classification-based modeling or prediction from binary (e.g., presence/absence of genes) or categorical (e.g., allele profiles) genomic data.
# Context
The scikit-learn (sklearn)-based Python workflow is inspired by an older caret-based R workflow (https://doi.org/10.1186/s12864-023-09667-w), independently supports both modeling (i.e., training and testing) and prediction (i.e., based on a pre-built model), and implements 5 feature selection methods, 20 model classifiers, hyperparameter tuning, performance metric computation, feature and permutation importance analyses, prediction probability estimation, execution monitoring via progress bars, and parallel processing.
# Version (release)
1.3.0 (December 2025)
# Dependencies
The Python script GenomicBasedClassification.py was prepared and tested with the Python version 3.12 and Ubuntu 20.04 LTS Focal Fossa.
- pandas==2.2.2
- imbalanced-learn==0.13.0
- scikit-learn==1.5.2
- xgboost==2.1.3
- numpy==1.26.4
- joblib==1.5.1
- tqdm==4.67.1
- tqdm-joblib==0.0.4
- catboost==1.2.8
# Implemented feature selection methods
- SelectKBest (SKB): https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.SelectKBest.html
- SelectFromModel with lasso-regularized logistic regression (laSFM): https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.SelectFromModel.html
- SelectFromModel with elasticnet-regularized logistic regression (enSFM): https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.SelectFromModel.html
- SelectFromModel with ridge-regularized logistic regression (riSFM): https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.SelectFromModel.html
- SelectFromModel with Random Forest (rfSFM): https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.SelectFromModel.html
# Implemented model classifiers
- adaboost (ADA): https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.AdaBoostClassifier.html
- catboost (CAT): https://catboost.ai/docs/en/concepts/python-reference_catboostclassifier
- decision tree (DT): https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html
- elasticnet-regularized logistic regression (EN): https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html
- extra trees (ET): https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.ExtraTreesClassifier.html
- gradient boosting (GB): https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingClassifier.html
- gaussian naive bayes (GNB): https://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.GaussianNB.html
- histogram-based gradient boosting (HGB): https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.HistGradientBoostingClassifier.html
- k-nearest neighbors (KNN): https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html
- lasso-regularized logistic regression (LA): https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html
- light gradient boosting machine (LGBM): https://lightgbm.readthedocs.io/en/latest/pythonapi/lightgbm.LGBMClassifier.html
- linear discriminant analysis (LDA): https://scikit-learn.org/stable/modules/generated/sklearn.discriminant_analysis.LinearDiscriminantAnalysis.html
- logistic regression (LR): https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html
- multi-layer perceptron (MLP): https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPClassifier.html
- nu support vector (NSV): https://scikit-learn.org/stable/modules/generated/sklearn.svm.NuSVC.html
- quadratic discriminant analysis (QDA): https://scikit-learn.org/stable/modules/generated/sklearn.discriminant_analysis.QuadraticDiscriminantAnalysis.html
- random forest (RF): https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html
- ridge-regularized logistic regression (RI): https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html
- support vector classification (SVC): https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html
- extreme gradient boosting (XGB): https://xgboost.readthedocs.io/en/stable/python/python_api.html#xgboost.XGBClassifier
# Recommended environments
## install Python libraries with pip
```
pip3.12 install pandas==2.2.2
pip3.12 install imbalanced-learn==0.13.0
pip3.12 install scikit-learn==1.5.2
pip3.12 install catboost==1.2.8
pip3.12 install lightgbm==4.6.0
pip3.12 install xgboost==2.1.3
pip3.12 install numpy==1.26.4
pip3.12 install joblib==1.5.1
pip3.12 install tqdm==4.67.1
pip3.12 install tqdm-joblib==0.0.4
```
## or install a Docker image
```
docker pull nicolasradomski/genomicbasedclassification:1.3.0
```
## or install a Conda environment
```
conda update --all
conda --version # conda 25.7.0
conda create --name env_conda_GenomicBasedClassification_1.3.0 python=3.12
conda activate env_conda_GenomicBasedClassification_1.3.0
python --version # Python 3.12.12
conda install -c conda-forge mamba=2.0.5
mamba install -c conda-forge pandas=2.2.2
mamba install -c conda-forge imbalanced-learn=0.13.0
mamba install -c conda-forge scikit-learn=1.5.2
mamba install -c conda-forge catboost=1.2.8
mamba install -c conda-forge lightgbm==4.6.0
mamba install -c conda-forge xgboost=2.1.3
mamba install -c conda-forge numpy=1.26.4
mamba install -c conda-forge joblib==1.5.1
mamba install -c conda-forge tqdm=4.67.1
mamba install -c nicolasradomski tqdm-joblib=0.0.4
conda list -n env_conda_GenomicBasedClassification_1.3.0
conda deactivate # after usage
```
## or install a Conda package
```
conda update --all
conda --version # conda 25.7.0
conda create -n env_anaconda_GenomicBasedClassification_1.3.0 -c nicolasradomski -c conda-forge -c defaults genomicbasedclassification=1.3.0
conda activate env_anaconda_GenomicBasedClassification_1.3.0
conda deactivate # after usage
```
# Helps
## modeling
```
usage: GenomicBasedClassification.py modeling [-h] -m INPUTPATH_MUTATIONS -ph INPUTPATH_PHENOTYPES [-da {random,manual}] [-sp SPLITTING] [-l LIMIT] [-fs FEATURESELECTION] [-c CLASSIFIER]
                                              [-k FOLD] [-pa PARAMETERS] [-j JOBS] [-pi] [-nr NREPEATS] [-o OUTPUTPATH] [-x PREFIX] [-di DIGITS] [-de DEBUG] [-w] [-nc]
options:
  -h, --help            show this help message and exit
  -m INPUTPATH_MUTATIONS, --mutations INPUTPATH_MUTATIONS
                        Absolute or relative input path of tab-separated values (tsv) file including profiles of mutations. First column: sample identifiers identical to those in the
                        input file of phenotypes and datasets (header: e.g., sample). Other columns: profiles of mutations (header: labels of mutations). [MANDATORY]
  -ph INPUTPATH_PHENOTYPES, --phenotypes INPUTPATH_PHENOTYPES
                        Absolute or relative input path of tab-separated values (tsv) file including profiles of phenotypes and datasets. First column: sample identifiers identical to
                        those in the input file of mutations (header: e.g., sample). Second column: categorical phenotype (header: e.g., phenotype). Third column: 'training' or 'testing'
                        dataset (header: e.g., dataset). [MANDATORY]
  -da {random,manual}, --dataset {random,manual}
                        Perform random (i.e., 'random') or manual (i.e., 'manual') splitting of training and testing datasets through the holdout method. [OPTIONAL, DEFAULT: 'random']
  -sp SPLITTING, --split SPLITTING
                        Percentage of random splitting to prepare the training dataset through the holdout method. [OPTIONAL, DEFAULT: None]
  -l LIMIT, --limit LIMIT
                        Recommended minimum of samples per class in both the training and testing datasets to reliably estimate performance metrics. [OPTIONAL, DEFAULT: 10]
  -fs FEATURESELECTION, --featureselection FEATURESELECTION
                        Acronym of the classification-compatible feature selection method to use: SelectKBest (SKB), SelectFromModel with lasso-regularized logistic regression (laSFM),
                        SelectFromModel with elasticNet-regularized logistic regression (enSFM), SelectFromModel with ridge-regularized logistic regression (riSFM), or SelectFromModel
                        with random forest (rfSFM). These methods are suitable for high-dimensional binary or categorical-encoded features. [OPTIONAL, DEFAULT: None]
  -c CLASSIFIER, --classifier CLASSIFIER
                        Acronym of the classifier to use among adaboost (ADA), catboost (CAT), decision tree (DT), elasticnet-regularized logistic regression (EN), extra trees (ET),
                        gradient boosting (GB), gaussian naive bayes (GNB), histogram-based gradient boosting (HGB), k-nearest neighbors (KNN), lasso-regularized logistic regression
                        (LA), light gradient boosting machine (LGBM), linear discriminant analysis (LDA), logistic regression (LR), multi-layer perceptron (MLP), nu support vector (NSV),
                        quadratic discriminant analysis (QDA), random forest (RF), ridge-regularized logistic regression (RI), support vector classification (SVC) or extreme gradient
                        boosting (XGB). [OPTIONAL, DEFAULT: XGB]
  -k FOLD, --fold FOLD  Value defining k-1 groups of samples used to train against one group of validation through the repeated k-fold cross-validation method. [OPTIONAL, DEFAULT: 5]
  -pa PARAMETERS, --parameters PARAMETERS
                        Absolute or relative input path of a text (txt) file including tuning parameters compatible with the param_grid argument of the GridSearchCV function. (OPTIONAL)
  -j JOBS, --jobs JOBS  Value defining the number of jobs to run in parallel compatible with the n_jobs argument of the GridSearchCV function. [OPTIONAL, DEFAULT: -1]
  -pi, --permutationimportance
                        Compute permutation importance, which can be computationally expensive, especially with many features and/or high repetition counts. [OPTIONAL, DEFAULT: False]
  -nr NREPEATS, --nrepeats NREPEATS
                        Number of repetitions per feature for permutation importance; higher values provide more stable estimates but increase runtime. [OPTIONAL, DEFAULT: 10]
  -o OUTPUTPATH, --output OUTPUTPATH
                        Output path. [OPTIONAL, DEFAULT: .]
  -x PREFIX, --prefix PREFIX
                        Prefix of output files. [OPTIONAL, DEFAULT: output]
  -di DIGITS, --digits DIGITS
                        Number of decimal digits to round numerical results (e.g., accuracy, importance, metrics). [OPTIONAL, DEFAULT: 6]
  -de DEBUG, --debug DEBUG
                        Traceback level when an error occurs. [OPTIONAL, DEFAULT: 0]
  -w, --warnings        Do not ignore warnings if you want to improve the script. [OPTIONAL, DEFAULT: False]
  -nc, --no-check       Do not check versions of Python and packages. [OPTIONAL, DEFAULT: False]
```
## prediction
```
usage: GenomicBasedClassification.py prediction [-h] -m INPUTPATH_MUTATIONS -f INPUTPATH_FEATURES -fe INPUTPATH_FEATURE_ENCODER -t INPUTPATH_MODEL [-ce INPUTPATH_CLASS_ENCODER]
                                                [-o OUTPUTPATH] [-x PREFIX] [-di DIGITS] [-de DEBUG] [-w] [-nc]
options:
  -h, --help            show this help message and exit
  -m INPUTPATH_MUTATIONS, --mutations INPUTPATH_MUTATIONS
                        Absolute or relative input path of a tab-separated values (tsv) file including profiles of mutations. First column: sample identifiers identical to those in the
                        input file of phenotypes and datasets (header: e.g., sample). Other columns: profiles of mutations (header: labels of mutations). [MANDATORY]
  -f INPUTPATH_FEATURES, --features INPUTPATH_FEATURES
                        Absolute or relative input path of an object (obj) file including features from the training dataset (i.e., mutations). [MANDATORY]
  -fe INPUTPATH_FEATURE_ENCODER, --featureencoder INPUTPATH_FEATURE_ENCODER
                        Absolute or relative input path of an object (obj) file including encoder from the training dataset (i.e., mutations). [MANDATORY]
  -t INPUTPATH_MODEL, --model INPUTPATH_MODEL
                        Absolute or relative input path of an object (obj) file including a trained scikit-learn model. [MANDATORY]
  -ce INPUTPATH_CLASS_ENCODER, --classencoder INPUTPATH_CLASS_ENCODER
                        Absolute or relative input path of an object (obj) file including trained scikit-learn class encoder (i.e., phenotypes) for the XGB model. [OPTIONAL]
  -o OUTPUTPATH, --output OUTPUTPATH
                        Absolute or relative output path. [OPTIONAL, DEFAULT: .]
  -x PREFIX, --prefix PREFIX
                        Prefix of output files. [OPTIONAL, DEFAULT: output_]
  -di DIGITS, --digits DIGITS
                        Number of decimal digits to round numerical results (e.g., accuracy, importance, metrics). [OPTIONAL, DEFAULT: 6]
  -de DEBUG, --debug DEBUG
                        Traceback level when an error occurs. [OPTIONAL, DEFAULT: 0]
  -w, --warnings        Do not ignore warnings if you want to improve the script. [OPTIONAL, DEFAULT: False]
  -nc, --no-check       Do not check versions of Python and packages. [OPTIONAL, DEFAULT: False]
```
# Expected input files
## phenotypes and datasets for modeling (e.g., phenotype_dataset.tsv)
```
sample    phenotype	dataset
S0.1.01   pig		training
S0.1.02   poultry	training
S0.1.03   poultry	training
S0.1.04   pig		training
S0.1.05   poultry	training
S0.1.06   poultry	training
S0.1.07   pig		testing
S0.1.08   poultry	training
S0.1.09   pig		testing
S0.1.10   pig		testing
```
## genomic data for modeling (e.g., genomic_profiles_for_modeling.tsv). "A" and "L" stand for alleles and locus, respectively.
```
sample		L_1	L_2	L_3	L_4	L_5	L_6	L_7	L_8	L_9	L_10
S0.1.01 	A3	A2	A3	A4	A5	A6	A7	A3	A4	A10
S0.1.02 	A8	A5	A3	A4	A5	A6	A7	A3	A4	A10
S0.1.03 	A6	A7	A6	A2	A17	A5	A6	A7	A8	A18
S0.1.04 	A12	A44	A8	A5	A16	A4	A5	A6	A12	A17
S0.1.05 	A6	A7	A15	A16	A3	A14	A6	A7	A8	A18
S0.1.06 	A6	A7	A15	A16	A8	A5	A6	A7	A8	A18
S0.1.07 	A7		A9	A10	A11	A14	A3	A2	A10	A16
S0.1.08 	A6	A7	A15	A16	A17	A5	A7	A5	A8	A18
S0.1.09 	A12	A13	A14	A15	A16	A4	A5	A6	A3	A2
S0.1.10 	A12	A13	A14	A15	A16	A4	A5	A6	A8	A8
```
## tuning parameters for modeling
### example with SKB as the feature selection method and XGB as the model regressor (tuning_parameters_XGB.txt)
```
{
# --- Feature selection (SelectKBest) ---
# used only to modify selected SKB behavior
'feature_selection__k': [25, 50], # number of top features to select based on univariate scores
'feature_selection__score_func': [mutual_info_classif], # scoring function estimating mutual information for classification tasks
# --- Model tuning (XGBClassifier) ---
# used only to modify XGB behavior
'model__max_depth': [3, 5], # tree depth
'model__eta': [0.1, 0.3], # learning rate
'model__n_estimators': [50, 100] # number of boosting rounds
}
```
### other examples depending on the selected feature selection method and model classifier
- tuning_parameters_ADA.txt
- tuning_parameters_CAT.txt
- tuning_parameters_DT.txt
- tuning_parameters_EN.txt
- tuning_parameters_ET.txt
- tuning_parameters_GB.txt
- tuning_parameters_GNB.txt
- tuning_parameters_HGB.txt
- tuning_parameters_KNN.txt
- tuning_parameters_LA.txt
- tuning_parameters_LDA.txt
- tuning_parameters_LGBM.txt
- tuning_parameters_LR.txt
- tuning_parameters_MLP.txt
- tuning_parameters_NSV.txt
- tuning_parameters_QDA.txt
- tuning_parameters_RF.txt
- tuning_parameters_RI.txt
- tuning_parameters_SVC.txt
- tuning_parameters_XGB.txt
## genomic profils for prediction (e.g., genomic_profiles_for_prediction.tsv). "A" and "L" stand for alleles and locus, respectively.
```
sample		L_1	L_2	L_3	L_4	L_5	L_6	L_7	L_8	L_9	L_10	L_11
S2.1.01 	A3	A2	A3	A4	A5	A6	A7	A3	A4	A1	A10
S2.1.02 	A8	A5	A3	A4	A5	A6	A7	A3	A4	A1	A10
S2.1.03 	A6	A7	A6	A2	A17	A5	A6	A7	A8	A1	A18
S2.1.04 	 	A13	A8	A5	A16	A4	A5	A6	A12	A1	A17
S2.1.05 	A6	A24	A15	A16	A3	A14	A6	A7	A8	A1	A18
S2.1.06 	A6	A7	A15	A16	A8	A5	A6	A7	A8	A1	A18
S2.1.07 	A7	A8	A9	A10	A11	A14	A3	A2	A88	A1	A16
S2.1.08 	A6	A7	A15	A16	A17	A5	A7	A5	A8	A1	A18
S2.1.09 	A12	A13	A14	A25	A16	A4	A5		A3	A1	A2
S2.1.10 	A12	A13	A14	A15	A16	A4	A5	A6	A8	A1	A8
```
# Examples of commands
## import the GitHub repository
```
git clone --branch v1.3.0 --single-branch https://github.com/Nicolas-Radomski/GenomicBasedClassification.git
cd GenomicBasedClassification
```
## using Python libraries from pip
### without feature selection and with the ADA model classifier
```
python3.12 GenomicBasedClassification.py modeling -m genomic_profiles_for_modeling.tsv -ph phenotype_dataset.tsv -o MyDirectory -x ADA_FirstAnalysis -da random -sp 80 -c ADA -k 5 -pa tuning_parameters_ADA.txt -pi -nr 5
python3.12 GenomicBasedClassification.py prediction -m genomic_profiles_for_prediction.tsv -t MyDirectory/ADA_FirstAnalysis_model.obj -f MyDirectory/ADA_FirstAnalysis_features.obj -fe MyDirectory/ADA_FirstAnalysis_feature_encoder.obj -o MyDirectory -x ADA_SecondAnalysis
```
### with the SKB feature selection and the CAT model classifier
```
python3.12 GenomicBasedClassification.py modeling -m genomic_profiles_for_modeling.tsv -ph MyDirectory/ADA_FirstAnalysis_phenotype_dataset.tsv -o MyDirectory -x CAT_FirstAnalysis -da manual -fs SKB -c CAT -k 5 -pa tuning_parameters_CAT.txt -pi -nr 5
python3.12 GenomicBasedClassification.py prediction -m genomic_profiles_for_prediction.tsv -t MyDirectory/CAT_FirstAnalysis_model.obj -f MyDirectory/CAT_FirstAnalysis_features.obj -fe MyDirectory/CAT_FirstAnalysis_feature_encoder.obj -o MyDirectory -x CAT_SecondAnalysis
```
### with the laSFM feature selection and the DT model classifier
```
python3.12 GenomicBasedClassification.py modeling -m genomic_profiles_for_modeling.tsv -ph MyDirectory/ADA_FirstAnalysis_phenotype_dataset.tsv -o MyDirectory -x DT_FirstAnalysis -da manual -fs laSFM -c DT -k 5 -pa tuning_parameters_DT.txt -pi -nr 5
python3.12 GenomicBasedClassification.py prediction -m genomic_profiles_for_prediction.tsv -t MyDirectory/DT_FirstAnalysis_model.obj -f MyDirectory/DT_FirstAnalysis_features.obj -fe MyDirectory/DT_FirstAnalysis_feature_encoder.obj -o MyDirectory -x DT_SecondAnalysis
```
### with the enSFM feature selection and the EN model classifier
```
python3.12 GenomicBasedClassification.py modeling -m genomic_profiles_for_modeling.tsv -ph MyDirectory/ADA_FirstAnalysis_phenotype_dataset.tsv -o MyDirectory -x EN_FirstAnalysis -da manual -fs enSFM -c EN -k 5 -pa tuning_parameters_EN.txt -pi -nr 5
python3.12 GenomicBasedClassification.py prediction -m genomic_profiles_for_prediction.tsv -t MyDirectory/EN_FirstAnalysis_model.obj -f MyDirectory/EN_FirstAnalysis_features.obj -fe MyDirectory/EN_FirstAnalysis_feature_encoder.obj -o MyDirectory -x EN_SecondAnalysis
```
### with the riSFM feature selection and the ET model classifier
```
python3.12 GenomicBasedClassification.py modeling -m genomic_profiles_for_modeling.tsv -ph MyDirectory/ADA_FirstAnalysis_phenotype_dataset.tsv -o MyDirectory -x ET_FirstAnalysis -da manual -fs riSFM -c ET -k 5 -pa tuning_parameters_ET.txt -pi -nr 5
python3.12 GenomicBasedClassification.py prediction -m genomic_profiles_for_prediction.tsv -t MyDirectory/ET_FirstAnalysis_model.obj -f MyDirectory/ET_FirstAnalysis_features.obj -fe MyDirectory/ET_FirstAnalysis_feature_encoder.obj -o MyDirectory -x ET_SecondAnalysis
```
### with the rfSFM feature selection and the GB model classifier
```
python3.12 GenomicBasedClassification.py modeling -m genomic_profiles_for_modeling.tsv -ph MyDirectory/ADA_FirstAnalysis_phenotype_dataset.tsv -o MyDirectory -x GB_FirstAnalysis -da manual -fs rfSFM -c GB -k 5 -pa tuning_parameters_GB.txt -pi -nr 5
python3.12 GenomicBasedClassification.py prediction -m genomic_profiles_for_prediction.tsv -t MyDirectory/GB_FirstAnalysis_model.obj -f MyDirectory/GB_FirstAnalysis_features.obj -fe MyDirectory/GB_FirstAnalysis_feature_encoder.obj -o MyDirectory -x GB_SecondAnalysis
```
### with the SKB feature selection and the GNB model classifier
```
python3.12 GenomicBasedClassification.py modeling -m genomic_profiles_for_modeling.tsv -ph MyDirectory/ADA_FirstAnalysis_phenotype_dataset.tsv -o MyDirectory -x GNB_FirstAnalysis -da manual -fs SKB -c GNB -k 5 -pa tuning_parameters_GNB.txt -pi -nr 5
python3.12 GenomicBasedClassification.py prediction -m genomic_profiles_for_prediction.tsv -t MyDirectory/GNB_FirstAnalysis_model.obj -f MyDirectory/GNB_FirstAnalysis_features.obj -fe MyDirectory/GNB_FirstAnalysis_feature_encoder.obj -o MyDirectory -x GNB_SecondAnalysis
```
### with the SKB feature selection and the HGB model classifier
```
python3.12 GenomicBasedClassification.py modeling -m genomic_profiles_for_modeling.tsv -ph MyDirectory/ADA_FirstAnalysis_phenotype_dataset.tsv -o MyDirectory -x HGB_FirstAnalysis -da manual -fs SKB -c HGB -k 5 -pa tuning_parameters_HGB.txt -pi -nr 5
python3.12 GenomicBasedClassification.py prediction -m genomic_profiles_for_prediction.tsv -t MyDirectory/HGB_FirstAnalysis_model.obj -f MyDirectory/HGB_FirstAnalysis_features.obj -fe MyDirectory/HGB_FirstAnalysis_feature_encoder.obj -o MyDirectory -x HGB_SecondAnalysis
```
### with the SKB feature selection and the KNN model classifier
```
python3.12 GenomicBasedClassification.py modeling -m genomic_profiles_for_modeling.tsv -ph MyDirectory/ADA_FirstAnalysis_phenotype_dataset.tsv -o MyDirectory -x KNN_FirstAnalysis -da manual -fs SKB -c KNN -k 5 -pa tuning_parameters_KNN.txt -pi -nr 5
python3.12 GenomicBasedClassification.py prediction -m genomic_profiles_for_prediction.tsv -t MyDirectory/KNN_FirstAnalysis_model.obj -f MyDirectory/KNN_FirstAnalysis_features.obj -fe MyDirectory/KNN_FirstAnalysis_feature_encoder.obj -o MyDirectory -x KNN_SecondAnalysis
```
### with the SKB feature selection and the LA model classifier
```
python3.12 GenomicBasedClassification.py modeling -m genomic_profiles_for_modeling.tsv -ph MyDirectory/ADA_FirstAnalysis_phenotype_dataset.tsv -o MyDirectory -x LA_FirstAnalysis -da manual -fs SKB -c LA -k 5 -pa tuning_parameters_LA.txt -pi -nr 5
python3.12 GenomicBasedClassification.py prediction -m genomic_profiles_for_prediction.tsv -t MyDirectory/LA_FirstAnalysis_model.obj -f MyDirectory/LA_FirstAnalysis_features.obj -fe MyDirectory/LA_FirstAnalysis_feature_encoder.obj -o MyDirectory -x LA_SecondAnalysis
```
### with the SKB feature selection and the LDA model classifier
```
python3.12 GenomicBasedClassification.py modeling -m genomic_profiles_for_modeling.tsv -ph MyDirectory/ADA_FirstAnalysis_phenotype_dataset.tsv -o MyDirectory -x LDA_FirstAnalysis -da manual -fs SKB -c LDA -k 5 -pa tuning_parameters_LDA.txt -pi -nr 5
python3.12 GenomicBasedClassification.py prediction -m genomic_profiles_for_prediction.tsv -t MyDirectory/LDA_FirstAnalysis_model.obj -f MyDirectory/LDA_FirstAnalysis_features.obj -fe MyDirectory/LDA_FirstAnalysis_feature_encoder.obj -o MyDirectory -x LDA_SecondAnalysis
```
### with the SKB feature selection and the LGBM model classifier
```
python3.12 GenomicBasedClassification.py modeling -m genomic_profiles_for_modeling.tsv -ph MyDirectory/ADA_FirstAnalysis_phenotype_dataset.tsv -o MyDirectory -x LGBM_FirstAnalysis -da manual -fs SKB -c LGBM -k 5 -pa tuning_parameters_LGBM.txt -pi -nr 5
python3.12 GenomicBasedClassification.py prediction -m genomic_profiles_for_prediction.tsv -t MyDirectory/LGBM_FirstAnalysis_model.obj -f MyDirectory/LGBM_FirstAnalysis_features.obj -fe MyDirectory/LGBM_FirstAnalysis_feature_encoder.obj -o MyDirectory -x LGBM_SecondAnalysis
```
### with the SKB feature selection and the LR model classifier
```
python3.12 GenomicBasedClassification.py modeling -m genomic_profiles_for_modeling.tsv -ph MyDirectory/ADA_FirstAnalysis_phenotype_dataset.tsv -o MyDirectory -x LR_FirstAnalysis -da manual -fs SKB -c LR -k 5 -pa tuning_parameters_LR.txt -pi -nr 5
python3.12 GenomicBasedClassification.py prediction -m genomic_profiles_for_prediction.tsv -t MyDirectory/LR_FirstAnalysis_model.obj -f MyDirectory/LR_FirstAnalysis_features.obj -fe MyDirectory/LR_FirstAnalysis_feature_encoder.obj -o MyDirectory -x LR_SecondAnalysis
```
### with the SKB feature selection and the MLP model classifier
```
python3.12 GenomicBasedClassification.py modeling -m genomic_profiles_for_modeling.tsv -ph MyDirectory/ADA_FirstAnalysis_phenotype_dataset.tsv -o MyDirectory -x MLP_FirstAnalysis -da manual -fs SKB -c MLP -k 5 -pa tuning_parameters_MLP.txt -pi -nr 5
python3.12 GenomicBasedClassification.py prediction -m genomic_profiles_for_prediction.tsv -t MyDirectory/MLP_FirstAnalysis_model.obj -f MyDirectory/MLP_FirstAnalysis_features.obj -fe MyDirectory/MLP_FirstAnalysis_feature_encoder.obj -o MyDirectory -x MLP_SecondAnalysis
```
### with the SKB feature selection and the NSV model classifier
```
python3.12 GenomicBasedClassification.py modeling -m genomic_profiles_for_modeling.tsv -ph MyDirectory/ADA_FirstAnalysis_phenotype_dataset.tsv -o MyDirectory -x NSV_FirstAnalysis -da manual -fs SKB -c NSV -k 5 -pa tuning_parameters_NSV.txt -pi -nr 5
python3.12 GenomicBasedClassification.py prediction -m genomic_profiles_for_prediction.tsv -t MyDirectory/NSV_FirstAnalysis_model.obj -f MyDirectory/NSV_FirstAnalysis_features.obj -fe MyDirectory/NSV_FirstAnalysis_feature_encoder.obj -o MyDirectory -x NSV_SecondAnalysis
```
### with the SKB feature selection and the QDA model classifier
```
python3.12 GenomicBasedClassification.py modeling -m genomic_profiles_for_modeling.tsv -ph MyDirectory/ADA_FirstAnalysis_phenotype_dataset.tsv -o MyDirectory -x QDA_FirstAnalysis -da manual -fs SKB -c QDA -k 5 -pa tuning_parameters_QDA.txt -pi -nr 5
python3.12 GenomicBasedClassification.py prediction -m genomic_profiles_for_prediction.tsv -t MyDirectory/QDA_FirstAnalysis_model.obj -f MyDirectory/QDA_FirstAnalysis_features.obj -fe MyDirectory/QDA_FirstAnalysis_feature_encoder.obj -o MyDirectory -x QDA_SecondAnalysis
```
### with the SKB feature selection and the RF model classifier
```
python3.12 GenomicBasedClassification.py modeling -m genomic_profiles_for_modeling.tsv -ph MyDirectory/ADA_FirstAnalysis_phenotype_dataset.tsv -o MyDirectory -x RF_FirstAnalysis -da manual -fs SKB -c RF -k 5 -pa tuning_parameters_RF.txt -pi -nr 5
python3.12 GenomicBasedClassification.py prediction -m genomic_profiles_for_prediction.tsv -t MyDirectory/RF_FirstAnalysis_model.obj -f MyDirectory/RF_FirstAnalysis_features.obj -fe MyDirectory/RF_FirstAnalysis_feature_encoder.obj -o MyDirectory -x RF_SecondAnalysis
```
### with the SKB feature selection and the RI model classifier
```
python3.12 GenomicBasedClassification.py modeling -m genomic_profiles_for_modeling.tsv -ph MyDirectory/ADA_FirstAnalysis_phenotype_dataset.tsv -o MyDirectory -x RI_FirstAnalysis -da manual -fs SKB -c RI -k 5 -pa tuning_parameters_RI.txt -pi -nr 5
python3.12 GenomicBasedClassification.py prediction -m genomic_profiles_for_prediction.tsv -t MyDirectory/RI_FirstAnalysis_model.obj -f MyDirectory/RI_FirstAnalysis_features.obj -fe MyDirectory/RI_FirstAnalysis_feature_encoder.obj -o MyDirectory -x RI_SecondAnalysis
```
### with the SKB feature selection and the SVC model classifier
```
python3.12 GenomicBasedClassification.py modeling -m genomic_profiles_for_modeling.tsv -ph MyDirectory/ADA_FirstAnalysis_phenotype_dataset.tsv -o MyDirectory -x SVC_FirstAnalysis -da manual -fs SKB -c SVC -k 5 -pa tuning_parameters_SVC.txt -pi -nr 5
python3.12 GenomicBasedClassification.py prediction -m genomic_profiles_for_prediction.tsv -t MyDirectory/SVC_FirstAnalysis_model.obj -f MyDirectory/SVC_FirstAnalysis_features.obj -fe MyDirectory/SVC_FirstAnalysis_feature_encoder.obj -o MyDirectory -x SVC_SecondAnalysis
```
### with the SKB feature selection and the XGB model classifier
```
python3.12 GenomicBasedClassification.py modeling -m genomic_profiles_for_modeling.tsv -ph MyDirectory/ADA_FirstAnalysis_phenotype_dataset.tsv -o MyDirectory -x XGB_FirstAnalysis -da manual -fs SKB -c XGB -k 5 -pa tuning_parameters_XGB.txt -pi -nr 5
python3.12 GenomicBasedClassification.py prediction -m genomic_profiles_for_prediction.tsv -t MyDirectory/XGB_FirstAnalysis_model.obj -f MyDirectory/XGB_FirstAnalysis_features.obj -fe MyDirectory/XGB_FirstAnalysis_feature_encoder.obj -ce MyDirectory/XGB_FirstAnalysis_class_encoder.obj -o MyDirectory -x XGB_SecondAnalysis
```
## using a Docker image
### without feature selection and with the ADA model classifier
```
docker run --rm --name nicolas -u $(id -u):$(id -g) -v $(pwd):/wd nicolasradomski/genomicbasedclassification:1.3.0 modeling -m genomic_profiles_for_modeling.tsv -ph phenotype_dataset.tsv -o MyDirectoryDockerHub -x ADA_FirstAnalysis -da random -sp 80 -c ADA -k 5 -pa tuning_parameters_ADA.txt -pi -nr 5
docker run --rm --name nicolas -u $(id -u):$(id -g) -v $(pwd):/wd nicolasradomski/genomicbasedclassification:1.3.0 prediction -m genomic_profiles_for_prediction.tsv -t MyDirectoryDockerHub/ADA_FirstAnalysis_model.obj -f MyDirectoryDockerHub/ADA_FirstAnalysis_features.obj -fe MyDirectoryDockerHub/ADA_FirstAnalysis_feature_encoder.obj -o MyDirectoryDockerHub -x ADA_SecondAnalysis
```
### with the SKB feature selection and the CAT model classifier
```
docker run --rm --name nicolas -u $(id -u):$(id -g) -v $(pwd):/wd nicolasradomski/genomicbasedclassification:1.3.0 modeling -m genomic_profiles_for_modeling.tsv -ph MyDirectoryDockerHub/ADA_FirstAnalysis_phenotype_dataset.tsv -o MyDirectoryDockerHub -x CAT_FirstAnalysis -da manual -fs SKB -c CAT -k 5 -pa tuning_parameters_CAT.txt -pi -nr 5
docker run --rm --name nicolas -u $(id -u):$(id -g) -v $(pwd):/wd nicolasradomski/genomicbasedclassification:1.3.0 prediction -m genomic_profiles_for_prediction.tsv -t MyDirectoryDockerHub/CAT_FirstAnalysis_model.obj -f MyDirectoryDockerHub/CAT_FirstAnalysis_features.obj -fe MyDirectoryDockerHub/CAT_FirstAnalysis_feature_encoder.obj -o MyDirectoryDockerHub -x CAT_SecondAnalysis
```
### with the laSFM feature selection and the DT model classifier
```
docker run --rm --name nicolas -u $(id -u):$(id -g) -v $(pwd):/wd nicolasradomski/genomicbasedclassification:1.3.0 modeling -m genomic_profiles_for_modeling.tsv -ph MyDirectoryDockerHub/ADA_FirstAnalysis_phenotype_dataset.tsv -o MyDirectoryDockerHub -x DT_FirstAnalysis -da manual -fs laSFM -c DT -k 5 -pa tuning_parameters_DT.txt -pi -nr 5
docker run --rm --name nicolas -u $(id -u):$(id -g) -v $(pwd):/wd nicolasradomski/genomicbasedclassification:1.3.0 prediction -m genomic_profiles_for_prediction.tsv -t MyDirectoryDockerHub/DT_FirstAnalysis_model.obj -f MyDirectoryDockerHub/DT_FirstAnalysis_features.obj -fe MyDirectoryDockerHub/DT_FirstAnalysis_feature_encoder.obj -o MyDirectoryDockerHub -x DT_SecondAnalysis
```
### with the enSFM feature selection and the EN model classifier
```
docker run --rm --name nicolas -u $(id -u):$(id -g) -v $(pwd):/wd nicolasradomski/genomicbasedclassification:1.3.0 modeling -m genomic_profiles_for_modeling.tsv -ph MyDirectoryDockerHub/ADA_FirstAnalysis_phenotype_dataset.tsv -o MyDirectoryDockerHub -x EN_FirstAnalysis -da manual -fs enSFM -c EN -k 5 -pa tuning_parameters_EN.txt -pi -nr 5
docker run --rm --name nicolas -u $(id -u):$(id -g) -v $(pwd):/wd nicolasradomski/genomicbasedclassification:1.3.0 prediction -m genomic_profiles_for_prediction.tsv -t MyDirectoryDockerHub/EN_FirstAnalysis_model.obj -f MyDirectoryDockerHub/EN_FirstAnalysis_features.obj -fe MyDirectoryDockerHub/EN_FirstAnalysis_feature_encoder.obj -o MyDirectoryDockerHub -x EN_SecondAnalysis
```
### with the riSFM feature selection and the ET model classifier
```
docker run --rm --name nicolas -u $(id -u):$(id -g) -v $(pwd):/wd nicolasradomski/genomicbasedclassification:1.3.0 modeling -m genomic_profiles_for_modeling.tsv -ph MyDirectoryDockerHub/ADA_FirstAnalysis_phenotype_dataset.tsv -o MyDirectoryDockerHub -x ET_FirstAnalysis -da manual -fs riSFM -c ET -k 5 -pa tuning_parameters_ET.txt -pi -nr 5
docker run --rm --name nicolas -u $(id -u):$(id -g) -v $(pwd):/wd nicolasradomski/genomicbasedclassification:1.3.0 prediction -m genomic_profiles_for_prediction.tsv -t MyDirectoryDockerHub/ET_FirstAnalysis_model.obj -f MyDirectoryDockerHub/ET_FirstAnalysis_features.obj -fe MyDirectoryDockerHub/ET_FirstAnalysis_feature_encoder.obj -o MyDirectoryDockerHub -x ET_SecondAnalysis
```
### with the rfSFM feature selection and the GB model classifier
```
docker run --rm --name nicolas -u $(id -u):$(id -g) -v $(pwd):/wd nicolasradomski/genomicbasedclassification:1.3.0 modeling -m genomic_profiles_for_modeling.tsv -ph MyDirectoryDockerHub/ADA_FirstAnalysis_phenotype_dataset.tsv -o MyDirectoryDockerHub -x GB_FirstAnalysis -da manual -fs rfSFM -c GB -k 5 -pa tuning_parameters_GB.txt -pi -nr 5
docker run --rm --name nicolas -u $(id -u):$(id -g) -v $(pwd):/wd nicolasradomski/genomicbasedclassification:1.3.0 prediction -m genomic_profiles_for_prediction.tsv -t MyDirectoryDockerHub/GB_FirstAnalysis_model.obj -f MyDirectoryDockerHub/GB_FirstAnalysis_features.obj -fe MyDirectoryDockerHub/GB_FirstAnalysis_feature_encoder.obj -o MyDirectoryDockerHub -x GB_SecondAnalysis
```
### with the SKB feature selection and the GNB model classifier
```
docker run --rm --name nicolas -u $(id -u):$(id -g) -v $(pwd):/wd nicolasradomski/genomicbasedclassification:1.3.0 modeling -m genomic_profiles_for_modeling.tsv -ph MyDirectoryDockerHub/ADA_FirstAnalysis_phenotype_dataset.tsv -o MyDirectoryDockerHub -x GNB_FirstAnalysis -da manual -fs SKB -c GNB -k 5 -pa tuning_parameters_GNB.txt -pi -nr 5
docker run --rm --name nicolas -u $(id -u):$(id -g) -v $(pwd):/wd nicolasradomski/genomicbasedclassification:1.3.0 prediction -m genomic_profiles_for_prediction.tsv -t MyDirectoryDockerHub/GNB_FirstAnalysis_model.obj -f MyDirectoryDockerHub/GNB_FirstAnalysis_features.obj -fe MyDirectoryDockerHub/GNB_FirstAnalysis_feature_encoder.obj -o MyDirectoryDockerHub -x GNB_SecondAnalysis
```
### with the SKB feature selection and the HGB model classifier
```
docker run --rm --name nicolas -u $(id -u):$(id -g) -v $(pwd):/wd nicolasradomski/genomicbasedclassification:1.3.0 modeling -m genomic_profiles_for_modeling.tsv -ph MyDirectoryDockerHub/ADA_FirstAnalysis_phenotype_dataset.tsv -o MyDirectoryDockerHub -x HGB_FirstAnalysis -da manual -fs SKB -c HGB -k 5 -pa tuning_parameters_HGB.txt -pi -nr 5
docker run --rm --name nicolas -u $(id -u):$(id -g) -v $(pwd):/wd nicolasradomski/genomicbasedclassification:1.3.0 prediction -m genomic_profiles_for_prediction.tsv -t MyDirectoryDockerHub/HGB_FirstAnalysis_model.obj -f MyDirectoryDockerHub/HGB_FirstAnalysis_features.obj -fe MyDirectoryDockerHub/HGB_FirstAnalysis_feature_encoder.obj -o MyDirectoryDockerHub -x HGB_SecondAnalysis
```
### with the SKB feature selection and the KNN model classifier
```
docker run --rm --name nicolas -u $(id -u):$(id -g) -v $(pwd):/wd nicolasradomski/genomicbasedclassification:1.3.0 modeling -m genomic_profiles_for_modeling.tsv -ph MyDirectoryDockerHub/ADA_FirstAnalysis_phenotype_dataset.tsv -o MyDirectoryDockerHub -x KNN_FirstAnalysis -da manual -fs SKB -c KNN -k 5 -pa tuning_parameters_KNN.txt -pi -nr 5
docker run --rm --name nicolas -u $(id -u):$(id -g) -v $(pwd):/wd nicolasradomski/genomicbasedclassification:1.3.0 prediction -m genomic_profiles_for_prediction.tsv -t MyDirectoryDockerHub/KNN_FirstAnalysis_model.obj -f MyDirectoryDockerHub/KNN_FirstAnalysis_features.obj -fe MyDirectoryDockerHub/KNN_FirstAnalysis_feature_encoder.obj -o MyDirectoryDockerHub -x KNN_SecondAnalysis
```
### with the SKB feature selection and the LA model classifier
```
docker run --rm --name nicolas -u $(id -u):$(id -g) -v $(pwd):/wd nicolasradomski/genomicbasedclassification:1.3.0 modeling -m genomic_profiles_for_modeling.tsv -ph MyDirectoryDockerHub/ADA_FirstAnalysis_phenotype_dataset.tsv -o MyDirectoryDockerHub -x LA_FirstAnalysis -da manual -fs SKB -c LA -k 5 -pa tuning_parameters_LA.txt -pi -nr 5
docker run --rm --name nicolas -u $(id -u):$(id -g) -v $(pwd):/wd nicolasradomski/genomicbasedclassification:1.3.0 prediction -m genomic_profiles_for_prediction.tsv -t MyDirectoryDockerHub/LA_FirstAnalysis_model.obj -f MyDirectoryDockerHub/LA_FirstAnalysis_features.obj -fe MyDirectoryDockerHub/LA_FirstAnalysis_feature_encoder.obj -o MyDirectoryDockerHub -x LA_SecondAnalysis
```
### with the SKB feature selection and the LDA model classifier
```
docker run --rm --name nicolas -u $(id -u):$(id -g) -v $(pwd):/wd nicolasradomski/genomicbasedclassification:1.3.0 modeling -m genomic_profiles_for_modeling.tsv -ph MyDirectoryDockerHub/ADA_FirstAnalysis_phenotype_dataset.tsv -o MyDirectoryDockerHub -x LDA_FirstAnalysis -da manual -fs SKB -c LDA -k 5 -pa tuning_parameters_LDA.txt -pi -nr 5
docker run --rm --name nicolas -u $(id -u):$(id -g) -v $(pwd):/wd nicolasradomski/genomicbasedclassification:1.3.0 prediction -m genomic_profiles_for_prediction.tsv -t MyDirectoryDockerHub/LDA_FirstAnalysis_model.obj -f MyDirectoryDockerHub/LDA_FirstAnalysis_features.obj -fe MyDirectoryDockerHub/LDA_FirstAnalysis_feature_encoder.obj -o MyDirectoryDockerHub -x LDA_SecondAnalysis
```
### with the SKB feature selection and the LGBM model classifier
```
docker run --rm --name nicolas -u $(id -u):$(id -g) -v $(pwd):/wd nicolasradomski/genomicbasedclassification:1.3.0 modeling -m genomic_profiles_for_modeling.tsv -ph MyDirectoryDockerHub/ADA_FirstAnalysis_phenotype_dataset.tsv -o MyDirectoryDockerHub -x LGBM_FirstAnalysis -da manual -fs SKB -c LGBM -k 5 -pa tuning_parameters_LGBM.txt -pi -nr 5
docker run --rm --name nicolas -u $(id -u):$(id -g) -v $(pwd):/wd nicolasradomski/genomicbasedclassification:1.3.0 prediction -m genomic_profiles_for_prediction.tsv -t MyDirectoryDockerHub/LGBM_FirstAnalysis_model.obj -f MyDirectoryDockerHub/LGBM_FirstAnalysis_features.obj -fe MyDirectoryDockerHub/LGBM_FirstAnalysis_feature_encoder.obj -o MyDirectoryDockerHub -x LGBM_SecondAnalysis
```
### with the SKB feature selection and the LR model classifier
```
docker run --rm --name nicolas -u $(id -u):$(id -g) -v $(pwd):/wd nicolasradomski/genomicbasedclassification:1.3.0 modeling -m genomic_profiles_for_modeling.tsv -ph MyDirectoryDockerHub/ADA_FirstAnalysis_phenotype_dataset.tsv -o MyDirectoryDockerHub -x LR_FirstAnalysis -da manual -fs SKB -c LR -k 5 -pa tuning_parameters_LR.txt -pi -nr 5
docker run --rm --name nicolas -u $(id -u):$(id -g) -v $(pwd):/wd nicolasradomski/genomicbasedclassification:1.3.0 prediction -m genomic_profiles_for_prediction.tsv -t MyDirectoryDockerHub/LR_FirstAnalysis_model.obj -f MyDirectoryDockerHub/LR_FirstAnalysis_features.obj -fe MyDirectoryDockerHub/LR_FirstAnalysis_feature_encoder.obj -o MyDirectoryDockerHub -x LR_SecondAnalysis
```
### with the SKB feature selection and the MLP model classifier
```
docker run --rm --name nicolas -u $(id -u):$(id -g) -v $(pwd):/wd nicolasradomski/genomicbasedclassification:1.3.0 modeling -m genomic_profiles_for_modeling.tsv -ph MyDirectoryDockerHub/ADA_FirstAnalysis_phenotype_dataset.tsv -o MyDirectoryDockerHub -x MLP_FirstAnalysis -da manual -fs SKB -c MLP -k 5 -pa tuning_parameters_MLP.txt -pi -nr 5
docker run --rm --name nicolas -u $(id -u):$(id -g) -v $(pwd):/wd nicolasradomski/genomicbasedclassification:1.3.0 prediction -m genomic_profiles_for_prediction.tsv -t MyDirectoryDockerHub/MLP_FirstAnalysis_model.obj -f MyDirectoryDockerHub/MLP_FirstAnalysis_features.obj -fe MyDirectoryDockerHub/MLP_FirstAnalysis_feature_encoder.obj -o MyDirectoryDockerHub -x MLP_SecondAnalysis
```
### with the SKB feature selection and the NSV model classifier
```
docker run --rm --name nicolas -u $(id -u):$(id -g) -v $(pwd):/wd nicolasradomski/genomicbasedclassification:1.3.0 modeling -m genomic_profiles_for_modeling.tsv -ph MyDirectoryDockerHub/ADA_FirstAnalysis_phenotype_dataset.tsv -o MyDirectoryDockerHub -x NSV_FirstAnalysis -da manual -fs SKB -c NSV -k 5 -pa tuning_parameters_NSV.txt -pi -nr 5
docker run --rm --name nicolas -u $(id -u):$(id -g) -v $(pwd):/wd nicolasradomski/genomicbasedclassification:1.3.0 prediction -m genomic_profiles_for_prediction.tsv -t MyDirectoryDockerHub/NSV_FirstAnalysis_model.obj -f MyDirectoryDockerHub/NSV_FirstAnalysis_features.obj -fe MyDirectoryDockerHub/NSV_FirstAnalysis_feature_encoder.obj -o MyDirectoryDockerHub -x NSV_SecondAnalysis
```
### with the SKB feature selection and the QDA model classifier
```
docker run --rm --name nicolas -u $(id -u):$(id -g) -v $(pwd):/wd nicolasradomski/genomicbasedclassification:1.3.0 modeling -m genomic_profiles_for_modeling.tsv -ph MyDirectoryDockerHub/ADA_FirstAnalysis_phenotype_dataset.tsv -o MyDirectoryDockerHub -x QDA_FirstAnalysis -da manual -fs SKB -c QDA -k 5 -pa tuning_parameters_QDA.txt -pi -nr 5
docker run --rm --name nicolas -u $(id -u):$(id -g) -v $(pwd):/wd nicolasradomski/genomicbasedclassification:1.3.0 prediction -m genomic_profiles_for_prediction.tsv -t MyDirectoryDockerHub/QDA_FirstAnalysis_model.obj -f MyDirectoryDockerHub/QDA_FirstAnalysis_features.obj -fe MyDirectoryDockerHub/QDA_FirstAnalysis_feature_encoder.obj -o MyDirectoryDockerHub -x QDA_SecondAnalysis
```
### with the SKB feature selection and the RF model classifier
```
docker run --rm --name nicolas -u $(id -u):$(id -g) -v $(pwd):/wd nicolasradomski/genomicbasedclassification:1.3.0 modeling -m genomic_profiles_for_modeling.tsv -ph MyDirectoryDockerHub/ADA_FirstAnalysis_phenotype_dataset.tsv -o MyDirectoryDockerHub -x RF_FirstAnalysis -da manual -fs SKB -c RF -k 5 -pa tuning_parameters_RF.txt -pi -nr 5
docker run --rm --name nicolas -u $(id -u):$(id -g) -v $(pwd):/wd nicolasradomski/genomicbasedclassification:1.3.0 prediction -m genomic_profiles_for_prediction.tsv -t MyDirectoryDockerHub/RF_FirstAnalysis_model.obj -f MyDirectoryDockerHub/RF_FirstAnalysis_features.obj -fe MyDirectoryDockerHub/RF_FirstAnalysis_feature_encoder.obj -o MyDirectoryDockerHub -x RF_SecondAnalysis
```
### with the SKB feature selection and the RI model classifier
```
docker run --rm --name nicolas -u $(id -u):$(id -g) -v $(pwd):/wd nicolasradomski/genomicbasedclassification:1.3.0 modeling -m genomic_profiles_for_modeling.tsv -ph MyDirectoryDockerHub/ADA_FirstAnalysis_phenotype_dataset.tsv -o MyDirectoryDockerHub -x RI_FirstAnalysis -da manual -fs SKB -c RI -k 5 -pa tuning_parameters_RI.txt -pi -nr 5
docker run --rm --name nicolas -u $(id -u):$(id -g) -v $(pwd):/wd nicolasradomski/genomicbasedclassification:1.3.0 prediction -m genomic_profiles_for_prediction.tsv -t MyDirectoryDockerHub/RI_FirstAnalysis_model.obj -f MyDirectoryDockerHub/RI_FirstAnalysis_features.obj -fe MyDirectoryDockerHub/RI_FirstAnalysis_feature_encoder.obj -o MyDirectoryDockerHub -x RI_SecondAnalysis
```
### with the SKB feature selection and the SVC model classifier
```
docker run --rm --name nicolas -u $(id -u):$(id -g) -v $(pwd):/wd nicolasradomski/genomicbasedclassification:1.3.0 modeling -m genomic_profiles_for_modeling.tsv -ph MyDirectoryDockerHub/ADA_FirstAnalysis_phenotype_dataset.tsv -o MyDirectoryDockerHub -x SVC_FirstAnalysis -da manual -fs SKB -c SVC -k 5 -pa tuning_parameters_SVC.txt -pi -nr 5
docker run --rm --name nicolas -u $(id -u):$(id -g) -v $(pwd):/wd nicolasradomski/genomicbasedclassification:1.3.0 prediction -m genomic_profiles_for_prediction.tsv -t MyDirectoryDockerHub/SVC_FirstAnalysis_model.obj -f MyDirectoryDockerHub/SVC_FirstAnalysis_features.obj -fe MyDirectoryDockerHub/SVC_FirstAnalysis_feature_encoder.obj -o MyDirectoryDockerHub -x SVC_SecondAnalysis
```
### with the SKB feature selection and the XGB model classifier
```
docker run --rm --name nicolas -u $(id -u):$(id -g) -v $(pwd):/wd nicolasradomski/genomicbasedclassification:1.3.0 modeling -m genomic_profiles_for_modeling.tsv -ph MyDirectoryDockerHub/ADA_FirstAnalysis_phenotype_dataset.tsv -o MyDirectoryDockerHub -x XGB_FirstAnalysis -da manual -fs SKB -c XGB -k 5 -pa tuning_parameters_XGB.txt -pi -nr 5
docker run --rm --name nicolas -u $(id -u):$(id -g) -v $(pwd):/wd nicolasradomski/genomicbasedclassification:1.3.0 prediction -m genomic_profiles_for_prediction.tsv -t MyDirectoryDockerHub/XGB_FirstAnalysis_model.obj -f MyDirectoryDockerHub/XGB_FirstAnalysis_features.obj -fe MyDirectoryDockerHub/XGB_FirstAnalysis_feature_encoder.obj -ce MyDirectoryDockerHub/XGB_FirstAnalysis_class_encoder.obj -o MyDirectoryDockerHub -x XGB_SecondAnalysis
```
## using a Conda environment
### without feature selection and with the ADA model classifier
```
python GenomicBasedClassification.py modeling -m genomic_profiles_for_modeling.tsv -ph phenotype_dataset.tsv -o MyDirectoryConda -x ADA_FirstAnalysis -da random -sp 80 -c ADA -k 5 -pa tuning_parameters_ADA.txt -pi -nr 5
python GenomicBasedClassification.py prediction -m genomic_profiles_for_prediction.tsv -t MyDirectoryConda/ADA_FirstAnalysis_model.obj -f MyDirectoryConda/ADA_FirstAnalysis_features.obj -fe MyDirectoryConda/ADA_FirstAnalysis_feature_encoder.obj -o MyDirectoryConda -x ADA_SecondAnalysis
```
### with the SKB feature selection and the CAT model classifier
```
python GenomicBasedClassification.py modeling -m genomic_profiles_for_modeling.tsv -ph MyDirectoryConda/ADA_FirstAnalysis_phenotype_dataset.tsv -o MyDirectoryConda -x CAT_FirstAnalysis -da manual -fs SKB -c CAT -k 5 -pa tuning_parameters_CAT.txt -pi -nr 5
python GenomicBasedClassification.py prediction -m genomic_profiles_for_prediction.tsv -t MyDirectoryConda/CAT_FirstAnalysis_model.obj -f MyDirectoryConda/CAT_FirstAnalysis_features.obj -fe MyDirectoryConda/CAT_FirstAnalysis_feature_encoder.obj -o MyDirectoryConda -x CAT_SecondAnalysis
```
### with the laSFM feature selection and the DT model classifier
```
python GenomicBasedClassification.py modeling -m genomic_profiles_for_modeling.tsv -ph MyDirectoryConda/ADA_FirstAnalysis_phenotype_dataset.tsv -o MyDirectoryConda -x DT_FirstAnalysis -da manual -fs laSFM -c DT -k 5 -pa tuning_parameters_DT.txt -pi -nr 5
python GenomicBasedClassification.py prediction -m genomic_profiles_for_prediction.tsv -t MyDirectoryConda/DT_FirstAnalysis_model.obj -f MyDirectoryConda/DT_FirstAnalysis_features.obj -fe MyDirectoryConda/DT_FirstAnalysis_feature_encoder.obj -o MyDirectoryConda -x DT_SecondAnalysis
```
### with the enSFM feature selection and the EN model classifier
```
python GenomicBasedClassification.py modeling -m genomic_profiles_for_modeling.tsv -ph MyDirectoryConda/ADA_FirstAnalysis_phenotype_dataset.tsv -o MyDirectoryConda -x EN_FirstAnalysis -da manual -fs enSFM -c EN -k 5 -pa tuning_parameters_EN.txt -pi -nr 5
python GenomicBasedClassification.py prediction -m genomic_profiles_for_prediction.tsv -t MyDirectoryConda/EN_FirstAnalysis_model.obj -f MyDirectoryConda/EN_FirstAnalysis_features.obj -fe MyDirectoryConda/EN_FirstAnalysis_feature_encoder.obj -o MyDirectoryConda -x EN_SecondAnalysis
```
### with the riSFM feature selection and the ET model classifier
```
python GenomicBasedClassification.py modeling -m genomic_profiles_for_modeling.tsv -ph MyDirectoryConda/ADA_FirstAnalysis_phenotype_dataset.tsv -o MyDirectoryConda -x ET_FirstAnalysis -da manual -fs riSFM -c ET -k 5 -pa tuning_parameters_ET.txt -pi -nr 5
python GenomicBasedClassification.py prediction -m genomic_profiles_for_prediction.tsv -t MyDirectoryConda/ET_FirstAnalysis_model.obj -f MyDirectoryConda/ET_FirstAnalysis_features.obj -fe MyDirectoryConda/ET_FirstAnalysis_feature_encoder.obj -o MyDirectoryConda -x ET_SecondAnalysis
```
### with the rfSFM feature selection and the GB model classifier
```
python GenomicBasedClassification.py modeling -m genomic_profiles_for_modeling.tsv -ph MyDirectoryConda/ADA_FirstAnalysis_phenotype_dataset.tsv -o MyDirectoryConda -x GB_FirstAnalysis -da manual -fs rfSFM -c GB -k 5 -pa tuning_parameters_GB.txt -pi -nr 5
python GenomicBasedClassification.py prediction -m genomic_profiles_for_prediction.tsv -t MyDirectoryConda/GB_FirstAnalysis_model.obj -f MyDirectoryConda/GB_FirstAnalysis_features.obj -fe MyDirectoryConda/GB_FirstAnalysis_feature_encoder.obj -o MyDirectoryConda -x GB_SecondAnalysis
```
### with the SKB feature selection and the GNB model classifier
```
python GenomicBasedClassification.py modeling -m genomic_profiles_for_modeling.tsv -ph MyDirectoryConda/ADA_FirstAnalysis_phenotype_dataset.tsv -o MyDirectoryConda -x GNB_FirstAnalysis -da manual -fs SKB -c GNB -k 5 -pa tuning_parameters_GNB.txt -pi -nr 5
python GenomicBasedClassification.py prediction -m genomic_profiles_for_prediction.tsv -t MyDirectoryConda/GNB_FirstAnalysis_model.obj -f MyDirectoryConda/GNB_FirstAnalysis_features.obj -fe MyDirectoryConda/GNB_FirstAnalysis_feature_encoder.obj -o MyDirectoryConda -x GNB_SecondAnalysis
```
### with the SKB feature selection and the HGB model classifier
```
python GenomicBasedClassification.py modeling -m genomic_profiles_for_modeling.tsv -ph MyDirectoryConda/ADA_FirstAnalysis_phenotype_dataset.tsv -o MyDirectoryConda -x HGB_FirstAnalysis -da manual -fs SKB -c HGB -k 5 -pa tuning_parameters_HGB.txt -pi -nr 5
python GenomicBasedClassification.py prediction -m genomic_profiles_for_prediction.tsv -t MyDirectoryConda/HGB_FirstAnalysis_model.obj -f MyDirectoryConda/HGB_FirstAnalysis_features.obj -fe MyDirectoryConda/HGB_FirstAnalysis_feature_encoder.obj -o MyDirectoryConda -x HGB_SecondAnalysis
```
### with the SKB feature selection and the KNN model classifier
```
python GenomicBasedClassification.py modeling -m genomic_profiles_for_modeling.tsv -ph MyDirectoryConda/ADA_FirstAnalysis_phenotype_dataset.tsv -o MyDirectoryConda -x KNN_FirstAnalysis -da manual -fs SKB -c KNN -k 5 -pa tuning_parameters_KNN.txt -pi -nr 5
python GenomicBasedClassification.py prediction -m genomic_profiles_for_prediction.tsv -t MyDirectoryConda/KNN_FirstAnalysis_model.obj -f MyDirectoryConda/KNN_FirstAnalysis_features.obj -fe MyDirectoryConda/KNN_FirstAnalysis_feature_encoder.obj -o MyDirectoryConda -x KNN_SecondAnalysis
```
### with the SKB feature selection and the LA model classifier
```
python GenomicBasedClassification.py modeling -m genomic_profiles_for_modeling.tsv -ph MyDirectoryConda/ADA_FirstAnalysis_phenotype_dataset.tsv -o MyDirectoryConda -x LA_FirstAnalysis -da manual -fs SKB -c LA -k 5 -pa tuning_parameters_LA.txt -pi -nr 5
python GenomicBasedClassification.py prediction -m genomic_profiles_for_prediction.tsv -t MyDirectoryConda/LA_FirstAnalysis_model.obj -f MyDirectoryConda/LA_FirstAnalysis_features.obj -fe MyDirectoryConda/LA_FirstAnalysis_feature_encoder.obj -o MyDirectoryConda -x LA_SecondAnalysis
```
### with the SKB feature selection and the LDA model classifier
```
python GenomicBasedClassification.py modeling -m genomic_profiles_for_modeling.tsv -ph MyDirectoryConda/ADA_FirstAnalysis_phenotype_dataset.tsv -o MyDirectoryConda -x LDA_FirstAnalysis -da manual -fs SKB -c LDA -k 5 -pa tuning_parameters_LDA.txt -pi -nr 5
python GenomicBasedClassification.py prediction -m genomic_profiles_for_prediction.tsv -t MyDirectoryConda/LDA_FirstAnalysis_model.obj -f MyDirectoryConda/LDA_FirstAnalysis_features.obj -fe MyDirectoryConda/LDA_FirstAnalysis_feature_encoder.obj -o MyDirectoryConda -x LDA_SecondAnalysis
```
### with the SKB feature selection and the LGBM model classifier
```
python GenomicBasedClassification.py modeling -m genomic_profiles_for_modeling.tsv -ph MyDirectoryConda/ADA_FirstAnalysis_phenotype_dataset.tsv -o MyDirectoryConda -x LGBM_FirstAnalysis -da manual -fs SKB -c LGBM -k 5 -pa tuning_parameters_LGBM.txt -pi -nr 5
python GenomicBasedClassification.py prediction -m genomic_profiles_for_prediction.tsv -t MyDirectoryConda/LGBM_FirstAnalysis_model.obj -f MyDirectoryConda/LGBM_FirstAnalysis_features.obj -fe MyDirectoryConda/LGBM_FirstAnalysis_feature_encoder.obj -o MyDirectoryConda -x LGBM_SecondAnalysis
```
### with the SKB feature selection and the LR model classifier
```
python GenomicBasedClassification.py modeling -m genomic_profiles_for_modeling.tsv -ph MyDirectoryConda/ADA_FirstAnalysis_phenotype_dataset.tsv -o MyDirectoryConda -x LR_FirstAnalysis -da manual -fs SKB -c LR -k 5 -pa tuning_parameters_LR.txt -pi -nr 5
python GenomicBasedClassification.py prediction -m genomic_profiles_for_prediction.tsv -t MyDirectoryConda/LR_FirstAnalysis_model.obj -f MyDirectoryConda/LR_FirstAnalysis_features.obj -fe MyDirectoryConda/LR_FirstAnalysis_feature_encoder.obj -o MyDirectoryConda -x LR_SecondAnalysis
```
### with the SKB feature selection and the MLP model classifier
```
python GenomicBasedClassification.py modeling -m genomic_profiles_for_modeling.tsv -ph MyDirectoryConda/ADA_FirstAnalysis_phenotype_dataset.tsv -o MyDirectoryConda -x MLP_FirstAnalysis -da manual -fs SKB -c MLP -k 5 -pa tuning_parameters_MLP.txt -pi -nr 5
python GenomicBasedClassification.py prediction -m genomic_profiles_for_prediction.tsv -t MyDirectoryConda/MLP_FirstAnalysis_model.obj -f MyDirectoryConda/MLP_FirstAnalysis_features.obj -fe MyDirectoryConda/MLP_FirstAnalysis_feature_encoder.obj -o MyDirectoryConda -x MLP_SecondAnalysis
```
### with the SKB feature selection and the NSV model classifier
```
python GenomicBasedClassification.py modeling -m genomic_profiles_for_modeling.tsv -ph MyDirectoryConda/ADA_FirstAnalysis_phenotype_dataset.tsv -o MyDirectoryConda -x NSV_FirstAnalysis -da manual -fs SKB -c NSV -k 5 -pa tuning_parameters_NSV.txt -pi -nr 5
python GenomicBasedClassification.py prediction -m genomic_profiles_for_prediction.tsv -t MyDirectoryConda/NSV_FirstAnalysis_model.obj -f MyDirectoryConda/NSV_FirstAnalysis_features.obj -fe MyDirectoryConda/NSV_FirstAnalysis_feature_encoder.obj -o MyDirectoryConda -x NSV_SecondAnalysis
```
### with the SKB feature selection and the QDA model classifier
```
python GenomicBasedClassification.py modeling -m genomic_profiles_for_modeling.tsv -ph MyDirectoryConda/ADA_FirstAnalysis_phenotype_dataset.tsv -o MyDirectoryConda -x QDA_FirstAnalysis -da manual -fs SKB -c QDA -k 5 -pa tuning_parameters_QDA.txt -pi -nr 5
python GenomicBasedClassification.py prediction -m genomic_profiles_for_prediction.tsv -t MyDirectoryConda/QDA_FirstAnalysis_model.obj -f MyDirectoryConda/QDA_FirstAnalysis_features.obj -fe MyDirectoryConda/QDA_FirstAnalysis_feature_encoder.obj -o MyDirectoryConda -x QDA_SecondAnalysis
```
### with the SKB feature selection and the RF model classifier
```
python GenomicBasedClassification.py modeling -m genomic_profiles_for_modeling.tsv -ph MyDirectoryConda/ADA_FirstAnalysis_phenotype_dataset.tsv -o MyDirectoryConda -x RF_FirstAnalysis -da manual -fs SKB -c RF -k 5 -pa tuning_parameters_RF.txt -pi -nr 5
python GenomicBasedClassification.py prediction -m genomic_profiles_for_prediction.tsv -t MyDirectoryConda/RF_FirstAnalysis_model.obj -f MyDirectoryConda/RF_FirstAnalysis_features.obj -fe MyDirectoryConda/RF_FirstAnalysis_feature_encoder.obj -o MyDirectoryConda -x RF_SecondAnalysis
```
### with the SKB feature selection and the RI model classifier
```
python GenomicBasedClassification.py modeling -m genomic_profiles_for_modeling.tsv -ph MyDirectoryConda/ADA_FirstAnalysis_phenotype_dataset.tsv -o MyDirectoryConda -x RI_FirstAnalysis -da manual -fs SKB -c RI -k 5 -pa tuning_parameters_RI.txt -pi -nr 5
python GenomicBasedClassification.py prediction -m genomic_profiles_for_prediction.tsv -t MyDirectoryConda/RI_FirstAnalysis_model.obj -f MyDirectoryConda/RI_FirstAnalysis_features.obj -fe MyDirectoryConda/RI_FirstAnalysis_feature_encoder.obj -o MyDirectoryConda -x RI_SecondAnalysis
```
### with the SKB feature selection and the SVC model classifier
```
python GenomicBasedClassification.py modeling -m genomic_profiles_for_modeling.tsv -ph MyDirectoryConda/ADA_FirstAnalysis_phenotype_dataset.tsv -o MyDirectoryConda -x SVC_FirstAnalysis -da manual -fs SKB -c SVC -k 5 -pa tuning_parameters_SVC.txt -pi -nr 5
python GenomicBasedClassification.py prediction -m genomic_profiles_for_prediction.tsv -t MyDirectoryConda/SVC_FirstAnalysis_model.obj -f MyDirectoryConda/SVC_FirstAnalysis_features.obj -fe MyDirectoryConda/SVC_FirstAnalysis_feature_encoder.obj -o MyDirectoryConda -x SVC_SecondAnalysis
```
### with the SKB feature selection and the XGB model classifier
```
python GenomicBasedClassification.py modeling -m genomic_profiles_for_modeling.tsv -ph MyDirectoryConda/ADA_FirstAnalysis_phenotype_dataset.tsv -o MyDirectoryConda -x XGB_FirstAnalysis -da manual -fs SKB -c XGB -k 5 -pa tuning_parameters_XGB.txt -pi -nr 5
python GenomicBasedClassification.py prediction -m genomic_profiles_for_prediction.tsv -t MyDirectoryConda/XGB_FirstAnalysis_model.obj -f MyDirectoryConda/XGB_FirstAnalysis_features.obj -fe MyDirectoryConda/XGB_FirstAnalysis_feature_encoder.obj -ce MyDirectoryConda/XGB_FirstAnalysis_class_encoder.obj -o MyDirectoryConda -x XGB_SecondAnalysis
```
## using a Conda package
### without feature selection and with the ADA model classifier
```
genomicbasedclassification modeling -m genomic_profiles_for_modeling.tsv -ph phenotype_dataset.tsv -o MyDirectoryAnaconda -x ADA_FirstAnalysis -da random -sp 80 -c ADA -k 5 -pa tuning_parameters_ADA.txt -pi -nr 5
genomicbasedclassification prediction -m genomic_profiles_for_prediction.tsv -t MyDirectoryAnaconda/ADA_FirstAnalysis_model.obj -f MyDirectoryAnaconda/ADA_FirstAnalysis_features.obj -fe MyDirectoryAnaconda/ADA_FirstAnalysis_feature_encoder.obj -o MyDirectoryAnaconda -x ADA_SecondAnalysis
```
### with the SKB feature selection and the CAT model classifier
```
genomicbasedclassification modeling -m genomic_profiles_for_modeling.tsv -ph MyDirectoryAnaconda/ADA_FirstAnalysis_phenotype_dataset.tsv -o MyDirectoryAnaconda -x CAT_FirstAnalysis -da manual -fs SKB -c CAT -k 5 -pa tuning_parameters_CAT.txt -pi -nr 5
genomicbasedclassification prediction -m genomic_profiles_for_prediction.tsv -t MyDirectoryAnaconda/CAT_FirstAnalysis_model.obj -f MyDirectoryAnaconda/CAT_FirstAnalysis_features.obj -fe MyDirectoryAnaconda/CAT_FirstAnalysis_feature_encoder.obj -o MyDirectoryAnaconda -x CAT_SecondAnalysis
```
### with the laSFM feature selection and the DT model classifier
```
genomicbasedclassification modeling -m genomic_profiles_for_modeling.tsv -ph MyDirectoryAnaconda/ADA_FirstAnalysis_phenotype_dataset.tsv -o MyDirectoryAnaconda -x DT_FirstAnalysis -da manual -fs laSFM -c DT -k 5 -pa tuning_parameters_DT.txt -pi -nr 5
genomicbasedclassification prediction -m genomic_profiles_for_prediction.tsv -t MyDirectoryAnaconda/DT_FirstAnalysis_model.obj -f MyDirectoryAnaconda/DT_FirstAnalysis_features.obj -fe MyDirectoryAnaconda/DT_FirstAnalysis_feature_encoder.obj -o MyDirectoryAnaconda -x DT_SecondAnalysis
```
### with the enSFM feature selection and the EN model classifier
```
genomicbasedclassification modeling -m genomic_profiles_for_modeling.tsv -ph MyDirectoryAnaconda/ADA_FirstAnalysis_phenotype_dataset.tsv -o MyDirectoryAnaconda -x EN_FirstAnalysis -da manual -fs enSFM -c EN -k 5 -pa tuning_parameters_EN.txt -pi -nr 5
genomicbasedclassification prediction -m genomic_profiles_for_prediction.tsv -t MyDirectoryAnaconda/EN_FirstAnalysis_model.obj -f MyDirectoryAnaconda/EN_FirstAnalysis_features.obj -fe MyDirectoryAnaconda/EN_FirstAnalysis_feature_encoder.obj -o MyDirectoryAnaconda -x EN_SecondAnalysis
```
### with the riSFM feature selection and the ET model classifier
```
genomicbasedclassification modeling -m genomic_profiles_for_modeling.tsv -ph MyDirectoryAnaconda/ADA_FirstAnalysis_phenotype_dataset.tsv -o MyDirectoryAnaconda -x ET_FirstAnalysis -da manual -fs riSFM -c ET -k 5 -pa tuning_parameters_ET.txt -pi -nr 5
genomicbasedclassification prediction -m genomic_profiles_for_prediction.tsv -t MyDirectoryAnaconda/ET_FirstAnalysis_model.obj -f MyDirectoryAnaconda/ET_FirstAnalysis_features.obj -fe MyDirectoryAnaconda/ET_FirstAnalysis_feature_encoder.obj -o MyDirectoryAnaconda -x ET_SecondAnalysis
```
### with the rfSFM feature selection and the GB model classifier
```
genomicbasedclassification modeling -m genomic_profiles_for_modeling.tsv -ph MyDirectoryAnaconda/ADA_FirstAnalysis_phenotype_dataset.tsv -o MyDirectoryAnaconda -x GB_FirstAnalysis -da manual -fs rfSFM -c GB -k 5 -pa tuning_parameters_GB.txt -pi -nr 5
genomicbasedclassification prediction -m genomic_profiles_for_prediction.tsv -t MyDirectoryAnaconda/GB_FirstAnalysis_model.obj -f MyDirectoryAnaconda/GB_FirstAnalysis_features.obj -fe MyDirectoryAnaconda/GB_FirstAnalysis_feature_encoder.obj -o MyDirectoryAnaconda -x GB_SecondAnalysis
```
### with the SKB feature selection and the GNB model classifier
```
genomicbasedclassification modeling -m genomic_profiles_for_modeling.tsv -ph MyDirectoryAnaconda/ADA_FirstAnalysis_phenotype_dataset.tsv -o MyDirectoryAnaconda -x GNB_FirstAnalysis -da manual -fs SKB -c GNB -k 5 -pa tuning_parameters_GNB.txt -pi -nr 5
genomicbasedclassification prediction -m genomic_profiles_for_prediction.tsv -t MyDirectoryAnaconda/GNB_FirstAnalysis_model.obj -f MyDirectoryAnaconda/GNB_FirstAnalysis_features.obj -fe MyDirectoryAnaconda/GNB_FirstAnalysis_feature_encoder.obj -o MyDirectoryAnaconda -x GNB_SecondAnalysis
```
### with the SKB feature selection and the HGB model classifier
```
genomicbasedclassification modeling -m genomic_profiles_for_modeling.tsv -ph MyDirectoryAnaconda/ADA_FirstAnalysis_phenotype_dataset.tsv -o MyDirectoryAnaconda -x HGB_FirstAnalysis -da manual -fs SKB -c HGB -k 5 -pa tuning_parameters_HGB.txt -pi -nr 5
genomicbasedclassification prediction -m genomic_profiles_for_prediction.tsv -t MyDirectoryAnaconda/HGB_FirstAnalysis_model.obj -f MyDirectoryAnaconda/HGB_FirstAnalysis_features.obj -fe MyDirectoryAnaconda/HGB_FirstAnalysis_feature_encoder.obj -o MyDirectoryAnaconda -x HGB_SecondAnalysis
```
### with the SKB feature selection and the KNN model classifier
```
genomicbasedclassification modeling -m genomic_profiles_for_modeling.tsv -ph MyDirectoryAnaconda/ADA_FirstAnalysis_phenotype_dataset.tsv -o MyDirectoryAnaconda -x KNN_FirstAnalysis -da manual -fs SKB -c KNN -k 5 -pa tuning_parameters_KNN.txt -pi -nr 5
genomicbasedclassification prediction -m genomic_profiles_for_prediction.tsv -t MyDirectoryAnaconda/KNN_FirstAnalysis_model.obj -f MyDirectoryAnaconda/KNN_FirstAnalysis_features.obj -fe MyDirectoryAnaconda/KNN_FirstAnalysis_feature_encoder.obj -o MyDirectoryAnaconda -x KNN_SecondAnalysis
```
### with the SKB feature selection and the LA model classifier
```
genomicbasedclassification modeling -m genomic_profiles_for_modeling.tsv -ph MyDirectoryAnaconda/ADA_FirstAnalysis_phenotype_dataset.tsv -o MyDirectoryAnaconda -x LA_FirstAnalysis -da manual -fs SKB -c LA -k 5 -pa tuning_parameters_LA.txt -pi -nr 5
genomicbasedclassification prediction -m genomic_profiles_for_prediction.tsv -t MyDirectoryAnaconda/LA_FirstAnalysis_model.obj -f MyDirectoryAnaconda/LA_FirstAnalysis_features.obj -fe MyDirectoryAnaconda/LA_FirstAnalysis_feature_encoder.obj -o MyDirectoryAnaconda -x LA_SecondAnalysis
```
### with the SKB feature selection and the LDA model classifier
```
genomicbasedclassification modeling -m genomic_profiles_for_modeling.tsv -ph MyDirectoryAnaconda/ADA_FirstAnalysis_phenotype_dataset.tsv -o MyDirectoryAnaconda -x LDA_FirstAnalysis -da manual -fs SKB -c LDA -k 5 -pa tuning_parameters_LDA.txt -pi -nr 5
genomicbasedclassification prediction -m genomic_profiles_for_prediction.tsv -t MyDirectoryAnaconda/LDA_FirstAnalysis_model.obj -f MyDirectoryAnaconda/LDA_FirstAnalysis_features.obj -fe MyDirectoryAnaconda/LDA_FirstAnalysis_feature_encoder.obj -o MyDirectoryAnaconda -x LDA_SecondAnalysis
```
### with the SKB feature selection and the LGBM model classifier
```
genomicbasedclassification modeling -m genomic_profiles_for_modeling.tsv -ph MyDirectoryAnaconda/ADA_FirstAnalysis_phenotype_dataset.tsv -o MyDirectoryAnaconda -x LGBM_FirstAnalysis -da manual -fs SKB -c LGBM -k 5 -pa tuning_parameters_LGBM.txt -pi -nr 5
genomicbasedclassification prediction -m genomic_profiles_for_prediction.tsv -t MyDirectoryAnaconda/LGBM_FirstAnalysis_model.obj -f MyDirectoryAnaconda/LGBM_FirstAnalysis_features.obj -fe MyDirectoryAnaconda/LGBM_FirstAnalysis_feature_encoder.obj -o MyDirectoryAnaconda -x LGBM_SecondAnalysis
```
### with the SKB feature selection and the LR model classifier
```
genomicbasedclassification modeling -m genomic_profiles_for_modeling.tsv -ph MyDirectoryAnaconda/ADA_FirstAnalysis_phenotype_dataset.tsv -o MyDirectoryAnaconda -x LR_FirstAnalysis -da manual -fs SKB -c LR -k 5 -pa tuning_parameters_LR.txt -pi -nr 5
genomicbasedclassification prediction -m genomic_profiles_for_prediction.tsv -t MyDirectoryAnaconda/LR_FirstAnalysis_model.obj -f MyDirectoryAnaconda/LR_FirstAnalysis_features.obj -fe MyDirectoryAnaconda/LR_FirstAnalysis_feature_encoder.obj -o MyDirectoryAnaconda -x LR_SecondAnalysis
```
### with the SKB feature selection and the MLP model classifier
```
genomicbasedclassification modeling -m genomic_profiles_for_modeling.tsv -ph MyDirectoryAnaconda/ADA_FirstAnalysis_phenotype_dataset.tsv -o MyDirectoryAnaconda -x MLP_FirstAnalysis -da manual -fs SKB -c MLP -k 5 -pa tuning_parameters_MLP.txt -pi -nr 5
genomicbasedclassification prediction -m genomic_profiles_for_prediction.tsv -t MyDirectoryAnaconda/MLP_FirstAnalysis_model.obj -f MyDirectoryAnaconda/MLP_FirstAnalysis_features.obj -fe MyDirectoryAnaconda/MLP_FirstAnalysis_feature_encoder.obj -o MyDirectoryAnaconda -x MLP_SecondAnalysis
```
### with the SKB feature selection and the NSV model classifier
```
genomicbasedclassification modeling -m genomic_profiles_for_modeling.tsv -ph MyDirectoryAnaconda/ADA_FirstAnalysis_phenotype_dataset.tsv -o MyDirectoryAnaconda -x NSV_FirstAnalysis -da manual -fs SKB -c NSV -k 5 -pa tuning_parameters_NSV.txt -pi -nr 5
genomicbasedclassification prediction -m genomic_profiles_for_prediction.tsv -t MyDirectoryAnaconda/NSV_FirstAnalysis_model.obj -f MyDirectoryAnaconda/NSV_FirstAnalysis_features.obj -fe MyDirectoryAnaconda/NSV_FirstAnalysis_feature_encoder.obj -o MyDirectoryAnaconda -x NSV_SecondAnalysis
```
### with the SKB feature selection and the QDA model classifier
```
genomicbasedclassification modeling -m genomic_profiles_for_modeling.tsv -ph MyDirectoryAnaconda/ADA_FirstAnalysis_phenotype_dataset.tsv -o MyDirectoryAnaconda -x QDA_FirstAnalysis -da manual -fs SKB -c QDA -k 5 -pa tuning_parameters_QDA.txt -pi -nr 5
genomicbasedclassification prediction -m genomic_profiles_for_prediction.tsv -t MyDirectoryAnaconda/QDA_FirstAnalysis_model.obj -f MyDirectoryAnaconda/QDA_FirstAnalysis_features.obj -fe MyDirectoryAnaconda/QDA_FirstAnalysis_feature_encoder.obj -o MyDirectoryAnaconda -x QDA_SecondAnalysis
```
### with the SKB feature selection and the RF model classifier
```
genomicbasedclassification modeling -m genomic_profiles_for_modeling.tsv -ph MyDirectoryAnaconda/ADA_FirstAnalysis_phenotype_dataset.tsv -o MyDirectoryAnaconda -x RF_FirstAnalysis -da manual -fs SKB -c RF -k 5 -pa tuning_parameters_RF.txt -pi -nr 5
genomicbasedclassification prediction -m genomic_profiles_for_prediction.tsv -t MyDirectoryAnaconda/RF_FirstAnalysis_model.obj -f MyDirectoryAnaconda/RF_FirstAnalysis_features.obj -fe MyDirectoryAnaconda/RF_FirstAnalysis_feature_encoder.obj -o MyDirectoryAnaconda -x RF_SecondAnalysis
```
### with the SKB feature selection and the RI model classifier
```
genomicbasedclassification modeling -m genomic_profiles_for_modeling.tsv -ph MyDirectoryAnaconda/ADA_FirstAnalysis_phenotype_dataset.tsv -o MyDirectoryAnaconda -x RI_FirstAnalysis -da manual -fs SKB -c RI -k 5 -pa tuning_parameters_RI.txt -pi -nr 5
genomicbasedclassification prediction -m genomic_profiles_for_prediction.tsv -t MyDirectoryAnaconda/RI_FirstAnalysis_model.obj -f MyDirectoryAnaconda/RI_FirstAnalysis_features.obj -fe MyDirectoryAnaconda/RI_FirstAnalysis_feature_encoder.obj -o MyDirectoryAnaconda -x RI_SecondAnalysis
```
### with the SKB feature selection and the SVC model classifier
```
genomicbasedclassification modeling -m genomic_profiles_for_modeling.tsv -ph MyDirectoryAnaconda/ADA_FirstAnalysis_phenotype_dataset.tsv -o MyDirectoryAnaconda -x SVC_FirstAnalysis -da manual -fs SKB -c SVC -k 5 -pa tuning_parameters_SVC.txt -pi -nr 5
genomicbasedclassification prediction -m genomic_profiles_for_prediction.tsv -t MyDirectoryAnaconda/SVC_FirstAnalysis_model.obj -f MyDirectoryAnaconda/SVC_FirstAnalysis_features.obj -fe MyDirectoryAnaconda/SVC_FirstAnalysis_feature_encoder.obj -o MyDirectoryAnaconda -x SVC_SecondAnalysis
```
### with the SKB feature selection and the XGB model classifier
```
genomicbasedclassification modeling -m genomic_profiles_for_modeling.tsv -ph MyDirectoryAnaconda/ADA_FirstAnalysis_phenotype_dataset.tsv -o MyDirectoryAnaconda -x XGB_FirstAnalysis -da manual -fs SKB -c XGB -k 5 -pa tuning_parameters_XGB.txt -pi -nr 5
genomicbasedclassification prediction -m genomic_profiles_for_prediction.tsv -t MyDirectoryAnaconda/XGB_FirstAnalysis_model.obj -f MyDirectoryAnaconda/XGB_FirstAnalysis_features.obj -fe MyDirectoryAnaconda/XGB_FirstAnalysis_feature_encoder.obj -ce MyDirectoryAnaconda/XGB_FirstAnalysis_class_encoder.obj -o MyDirectoryAnaconda -x XGB_SecondAnalysis
```
# Examples of expected output (see inclosed directory called 'MyDirectory')
## feature importance
```
 feature  importance
  L_3_A9    0.280788
  L_1_A1    0.209121
L_10_A13    0.124613
  L_1_A6    0.068516
L_10_A19    0.055184
 L_1_A12    0.050385
  L_1_A3    0.043766
 L_3_A16    0.037061
  L_4_A7    0.027436
L_10_A10    0.025891
  L_9_A8    0.014505
  L_5_A3    0.013375
  L_6_A5    0.011942
 L_5_A17    0.009594
L_10_A17    0.008494
  L_3_A6    0.004544
 L_4_A15    0.003981
  L_8_A8    0.002773
L_10_A18    0.002260
  L_8_A3    0.001830
```
## permutation importance
```
 feature  train_mean  train_std  test_mean  test_std
  L_2_A5    0.577217   0.076507   0.512612  0.061532
L_10_A13    0.414050   0.027764   0.630737  0.121266
 L_10_A2    0.252297   0.027889   0.166289  0.046921
 L_8_A13    0.146119   0.032002   0.166338  0.046689
  L_1_A4    0.137840   0.014877   0.067364  0.003423
L_10_A10    0.123660   0.007979   0.104844  0.005394
  L_1_A6    0.116510   0.017659   0.239731  0.089779
 L_9_A12    0.096977   0.012581   0.106619  0.007037
  L_6_A4    0.018595   0.003072  -0.002285  0.021886
  L_5_A3    0.009856   0.001290   0.016842  0.000608
 L_5_A11    0.006716   0.007499   0.002086  0.001323
  L_3_A9    0.005745   0.002412  -0.001259  0.004364
  L_3_A3    0.003716   0.000559   0.001306  0.001089
 L_2_A13    0.002325   0.000527   0.002042  0.000589
L_10_A19    0.001733   0.002360  -0.002564  0.003497
 L_5_A16    0.000335   0.000144   0.001344  0.000700
  L_8_A7    0.000028   0.000244  -0.001213  0.001120
  L_8_A2    0.000000   0.000000   0.000000  0.000000
  L_6_A9    0.000000   0.000000   0.000000  0.000000
 L_7_A10    0.000000   0.000000   0.000000  0.000000
```
## confusion matrix
```
from the training dataset: 
phenotype  fruit  pig  poultry
    fruit     48    4        4
      pig      0   56        0
  poultry      0    3       45
from the testing dataset: 
phenotype  fruit  pig  poultry
    fruit     12    1        1
      pig      0   14        0
  poultry      0    2       10
```
## metrics per class
```
from the training dataset: 
phenotype  TN  FP  FN  TP  support  accuracy  sensitivity  specificity  precision   recall  f1-score      MCC  ROC-AUC   PR-AUC  PRG-AUC  PRG-AUC_clipped
    fruit 104   0   8  48       56   0.95000     0.857143     1.000000   1.000000 0.857143  0.923077 0.892143 0.993990 0.985278 0.975663         0.975663
      pig  97   7   0  56       56   0.95625     1.000000     0.932692   0.888889 1.000000  0.941177 0.910527 0.994420 0.985231 0.975916         0.975916
  poultry 108   4   3  45       48   0.95625     0.937500     0.964286   0.918367 0.937500  0.927835 0.896548 0.993397 0.980439 0.972839         0.972839
from the testing dataset: 
phenotype  TN  FP  FN  TP  support  accuracy  sensitivity  specificity  precision   recall  f1-score      MCC  ROC-AUC   PR-AUC  PRG-AUC  PRG-AUC_clipped
    fruit  26   0   2  12       14     0.950     0.857143     1.000000   1.000000 0.857143  0.923077 0.892143 0.991758 0.979025 0.969140         0.969140
      pig  23   3   0  14       14     0.925     1.000000     0.884615   0.823529 1.000000  0.903226 0.853526 0.979396 0.962185 0.916234         0.916234
  poultry  27   1   2  10       12     0.925     0.833333     0.964286   0.909091 0.833333  0.869565 0.818596 0.992560 0.979604 0.963727         0.963727
```
## global metrics
```
from the training dataset: 
 accuracy  sensitivity  specificity  precision   recall  f1-score     MCC  Cohen kappa  ROC-AUC   PR-AUC  PRG-AUC  PRG-AUC_clipped
  0.93125     0.931548     0.965659   0.935752 0.931548  0.930696 0.89968     0.896665 0.993936 0.983649      0.5              0.5
from the testing dataset: 
 accuracy  sensitivity  specificity  precision   recall  f1-score      MCC  Cohen kappa  ROC-AUC   PR-AUC  PRG-AUC  PRG-AUC_clipped
      0.9     0.896825     0.949634   0.910873 0.896825  0.898623 0.855007     0.849341 0.987904 0.973605      0.5              0.5
```
## prediction during the modeling subcommand
```
 sample expectation prediction    fruit      pig  poultry
S0.1.02     poultry        pig 0.008989 0.872090 0.118921
S0.1.03     poultry    poultry 0.002700 0.000746 0.996554
S0.1.04         pig        pig 0.001595 0.997927 0.000478
S0.1.05     poultry    poultry 0.492025 0.003969 0.504006
S0.1.06     poultry    poultry 0.002710 0.000796 0.996495
S0.1.07         pig        pig 0.013854 0.980189 0.005957
S0.1.08     poultry    poultry 0.002540 0.000746 0.996714
S0.1.09         pig        pig 0.003541 0.995948 0.000511
S0.1.10         pig        pig 0.490655 0.504203 0.005142
S0.1.11     poultry    poultry 0.002540 0.000746 0.996714
S0.1.12         pig        pig 0.002056 0.997467 0.000477
S0.1.13     poultry    poultry 0.003566 0.038541 0.957893
S0.1.14       fruit      fruit 0.991943 0.003883 0.004174
S0.1.15       fruit      fruit 0.987030 0.007125 0.005845
S0.1.16       fruit        pig 0.167899 0.605344 0.226757
S0.1.17       fruit      fruit 0.982114 0.013118 0.004767
S0.1.18       fruit      fruit 0.983042 0.010897 0.006062
S0.1.19       fruit      fruit 0.992533 0.003311 0.004156
S1.1.01         pig        pig 0.004817 0.952642 0.042541
S1.1.02         pig        pig 0.008989 0.872090 0.118921
```
## prediction during the prediction subcommand
```
 sample prediction    fruit      pig  poultry
S2.1.01    poultry 0.060538 0.404823 0.534639
S2.1.02    poultry 0.063465 0.096945 0.839590
S2.1.03    poultry 0.002700 0.000746 0.996554
S2.1.04        pig 0.214195 0.727114 0.058691
S2.1.05      fruit 0.625629 0.004658 0.369714
S2.1.06    poultry 0.003031 0.000890 0.996079
S2.1.07        pig 0.013854 0.980189 0.005957
S2.1.08    poultry 0.002540 0.000746 0.996714
S2.1.09        pig 0.001709 0.997780 0.000512
S2.1.10        pig 0.490655 0.504203 0.005142
S2.1.11    poultry 0.002540 0.000746 0.996714
S2.1.12        pig 0.002202 0.997286 0.000511
S2.1.13    poultry 0.003688 0.005855 0.990457
S2.1.14      fruit 0.476902 0.382373 0.140725
S2.1.15        pig 0.070786 0.700570 0.228644
S2.1.16        pig 0.013485 0.840287 0.146228
S2.1.17        pig 0.085661 0.795675 0.118664
S2.1.18        pig 0.051124 0.776924 0.171952
S2.1.19      fruit 0.659991 0.169251 0.170758
S2.1.20        pig 0.070786 0.700570 0.228644
```
# Illustration
![workflow figure](https://github.com/Nicolas-Radomski/GenomicBasedClassification/blob/main/illustration.png)
# Funding
Ricerca Corrente - IZS AM 06/24 RC: "genomic data-based machine learning to predict categorical and continuous phenotypes by classification and regression".
# Acknowledgements
Many thanks to Andrea De Ruvo, Adriano Di Pasquale and ChatGPT for the insightful discussions that helped improve the algorithm.
# Reference
Pierluigi Castelli, Andrea De Ruvo, Andrea Bucciacchio, Nicola D'Alterio, Cesare Camma, Adriano Di Pasquale and Nicolas Radomski (2023) Harmonization of supervised machine learning practices for efficient source attribution of Listeria monocytogenes based on genomic data. 2023, BMC Genomics, 24(560):1-19, https://doi.org/10.1186/s12864-023-09667-w
# Repositories
- GitHub: https://github.com/Nicolas-Radomski/GenomicBasedClassification
- Docker Hub: https://hub.docker.com/r/nicolasradomski/genomicbasedclassification
- Anaconda Hub: https://anaconda.org/nicolasradomski/genomicbasedclassification
- R users: https://github.com/Nicolas-Radomski/GenomicBasedMachineLearning
# Author
Nicolas Radomski
