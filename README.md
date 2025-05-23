# Usage
The repository GenomicBasedClassification provides a Python (recommended version 3.12) script called GenomicBasedClassification.py to perform classification-based modeling or prediction from binary (e.g. presence/absence of genes) or categorical (e.g. allele profiles) genomic data.
# Context
The scikit-learn (sklearn) Python library-based workflow was inspired by an older caret R library-based version (https://doi.org/10.1186/s12864-023-09667-w), incorporating the independence of modeling (i.e. training and testing) and prediction (i.e. based on a pre-built model), the management of classification parameters, and the estimation of probabilities associated with predictions.
# Version (release)
1.1.0 (May 2025)
# Dependencies
The Python script GenomicBasedClassification.py was prepared and tested with the Python version 3.12 and Ubuntu 20.04 LTS Focal Fossa.
- pandas==2.2.2
- imbalanced-learn==0.13.0
- scikit-learn==1.5.2
- xgboost==2.1.3
- numpy==1.26.4
- joblib==1.4.2
- tqdm==4.67.1
- tqdm-joblib==0.0.4
# Implemented classifiers
- decision tree classifier (DT)
- k-nearest neighbors (KNN)
- logistic regression (LR)
- random forest (RF)
- support vector classification (SVC)
- extreme gradient boosting (XGB)
# Recommended environments
## install Python libraries with pip
```
pip3.12 install pandas==2.2.2
pip3.12 install imbalanced-learn==0.13.0
pip3.12 install scikit-learn==1.5.2
pip3.12 install xgboost==2.1.3
pip3.12 install numpy==1.26.4
pip3.12 install joblib==1.4.2
pip3.12 install tqdm==4.67.1
pip3.12 install tqdm-joblib==0.0.4
```
## or install a Docker image
```
docker pull nicolasradomski/genomicbasedclassification:1.1.0
```
## or install a Conda environment
```
conda update --all
conda create --name env_conda_GenomicBasedClassification_1.1.0 python=3.12
conda activate env_conda_GenomicBasedClassification_1.1.0
python --version
conda install -c conda-forge mamba=2.0.5
mamba install -c conda-forge pandas=2.2.2
mamba install -c conda-forge imbalanced-learn=0.13.0
mamba install -c conda-forge scikit-learn=1.5.2
mamba install -c conda-forge xgboost=2.1.3
mamba install -c conda-forge numpy=1.26.4
mamba install -c conda-forge joblib==1.4.2
mamba install -c conda-forge tqdm=4.67.1
mamba install -c nicolasradomski tqdm-joblib=0.0.4
conda list -n env_conda_GenomicBasedClassification_1.1.0
conda deactivate # after usage
```
## or install a Conda package
```
conda update --all
conda create -n env_anaconda_GenomicBasedClassification_1.1.0 -c nicolasradomski -c conda-forge -c defaults genomicbasedclassification=1.1.0
conda activate env_anaconda_GenomicBasedClassification_1.1.0
conda deactivate # after usage
```
# Helps
## modeling
```
usage: GenomicBasedClassification.py modeling [-h] -m INPUTPATH_MUTATIONS -ph INPUTPATH_PHENOTYPES [-da {random,manual}] [-s SPLITTING] [-l LIMIT] [-c CLASSIFIER]
                                                  [-k FOLD] [-pa PARAMETERS] [-j JOBS] [-o OUTPUTPATH] [-x PREFIX] [-de DEBUG] [-w] [-nc]
options:
  -h, --help            show this help message and exit
  -m INPUTPATH_MUTATIONS, --mutations INPUTPATH_MUTATIONS
                        Absolute or relative input path of tab-separated values (tsv) file including profiles of mutations. First column: sample identifiers identical to those
                        in the input file of phenotypes and datasets (header: e.g. sample). Other columns: profiles of mutations (header: labels of mutations). [MANDATORY]
  -ph INPUTPATH_PHENOTYPES, --phenotypes INPUTPATH_PHENOTYPES
                        Absolute or relative input path of tab-separated values (tsv) file including profiles of phenotypes and datasets. First column: sample identifiers
                        identical to those in the input file of mutations (header: e.g. sample). Second column: categorical phenotype (header: e.g. phenotype). Third column:
                        'training' or 'testing' dataset (header: e.g. dataset). [MANDATORY]
  -da {random,manual}, --dataset {random,manual}
                        Perform random (i.e. 'random') or manual (i.e. 'manual') splitting of training and testing datasets through the holdout method. [OPTIONAL, DEFAULT: 'random']
  -s SPLITTING, --split SPLITTING
                        Percentage of random splitting to prepare the training dataset through the holdout method. [OPTIONAL, DEFAULT: None]
  -l LIMIT, --limit LIMIT
                        Minimum number of samples per class required to estimate metrics from the training and testing datasets. [OPTIONAL, DEFAULT: 10]
  -c CLASSIFIER, --classifier CLASSIFIER
                        Acronym of the classifier to use among decision tree classifier (DT), k-nearest neighbors (KNN), logistic regression (LR), random forest (RF), support
                        vector classification (SVC) or extreme gradient boosting (XGB). [OPTIONAL, DEFAULT: XGB]
  -k FOLD, --fold FOLD  Value defining k-1 groups of samples used to train against one group of validation through the repeated k-fold cross-validation method. [OPTIONAL, DEFAULT: 5]
  -pa PARAMETERS, --parameters PARAMETERS
                        Absolute or relative input path of a text (txt) file including tuning parameters compatible with the param_grid argument of the GridSearchCV function.
                        (OPTIONAL)
  -j JOBS, --jobs JOBS  Value defining the number of jobs to run in parallel compatible with the n_jobs argument of the GridSearchCV function. [OPTIONAL, DEFAULT: -1]
  -o OUTPUTPATH, --output OUTPUTPATH
                        Output path. [OPTIONAL, DEFAULT: .]
  -x PREFIX, --prefix PREFIX
                        Prefix of output files. [OPTIONAL, DEFAULT: output]
  -de DEBUG, --debug DEBUG
                        Traceback level when an error occurs. [OPTIONAL, DEFAULT: 0]
  -w, --warnings        Do not ignore warnings if you want to improve the script. [OPTIONAL, DEFAULT: False]
  -nc, --no-check       Do not check versions of Python and packages. [OPTIONAL, DEFAULT: False]
```
## prediction
```
usage: GenomicBasedClassification.py prediction [-h] -m INPUTPATH_MUTATIONS -t INPUTPATH_MODEL -f INPUTPATH_FEATURES -ef INPUTPATH_ENCODED_FEATURES [-ec INPUTPATH_ENCODED_CLASSES]
                                                    [-o OUTPUTPATH] [-x PREFIX] [-de DEBUG] [-w] [-nc]
options:
  -h, --help            show this help message and exit
  -m INPUTPATH_MUTATIONS, --mutations INPUTPATH_MUTATIONS
                        Absolute or relative input path of a tab-separated values (tsv) file including profiles of mutations. First column: sample identifiers identical to those in the input file
                        of phenotypes and datasets (header: e.g. sample). Other columns: profiles of mutations (header: labels of mutations). [MANDATORY]
  -t INPUTPATH_MODEL, --model INPUTPATH_MODEL
                        Absolute or relative input path of an object (obj) file including a trained scikit-learn model. [MANDATORY]
  -f INPUTPATH_FEATURES, --features INPUTPATH_FEATURES
                        Absolute or relative input path of an object (obj) file including trained scikit-learn features (i.e. mutations). [MANDATORY]
  -ef INPUTPATH_ENCODED_FEATURES, --encodedfeatures INPUTPATH_ENCODED_FEATURES
                        Absolute or relative input path of an object (obj) file including trained scikit-learn encoded features (i.e. mutations). [MANDATORY]
  -ec INPUTPATH_ENCODED_CLASSES, --encodedclasses INPUTPATH_ENCODED_CLASSES
                        Absolute or relative input path of an object (obj) file including trained scikit-learn encoded classes (i.e. phenotypes) for the XGB model. [OPTIONAL]
  -o OUTPUTPATH, --output OUTPUTPATH
                        Absolute or relative output path. [OPTIONAL, DEFAULT: .]
  -x PREFIX, --prefix PREFIX
                        Prefix of output files. [OPTIONAL, DEFAULT: output_]
  -de DEBUG, --debug DEBUG
                        Traceback level when an error occurs. [OPTIONAL, DEFAULT: 0]
  -w, --warnings        Do not ignore warnings if you want to improve the script. [OPTIONAL, DEFAULT: False]
  -nc, --no-check       Do not check versions of Python and packages. [OPTIONAL, DEFAULT: False]
```
# Expected input files
## phenotypes and datasets for modeling (e.g. phenotype_dataset.tsv)
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
## genomic data for modeling (e.g. genomic_profiles_for_modeling.tsv). "A" and "L" stand for alleles and locus, respectively.
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
### for the DT classifier (tuning_parameters_DT.txt)
https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html
```
{
'criterion': ['gini', 'entropy'], # criteria to split the tree (Gini impurity or Entropy)
'splitter': ['best', 'random'], # splitter for deciding how to split at each node
'max_depth': [None, 5, 10, 15, 20], # maximum depth of the tree
'min_samples_split': [2, 5, 10], # minimum number of samples required to split an internal node
'min_samples_leaf': [1, 5, 10], # minimum number of samples required to be at a leaf node
'max_features': ['sqrt', 'log2', None], # number of features to consider at each split
'max_leaf_nodes': [None, 10, 20], # maximum number of leaf nodes
'random_state': [42] # for reproducibility
}
```
### for the KNN classifier (tuning_parameters_KNN.txt)
https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html
```
{
'n_neighbors': list(range(7, 21)), # keep the number of neighbors low
'algorithm': ['auto'], # try 'auto' for simplicity
'metric': ['minkowski'], # use Minkowski distance (default)
'leaf_size': [30], # restrict leaf size range
}
```
### for the LR classifier (tuning_parameters_LR.txt)
https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html
```
{
'solver': ['saga'], # saga is the only solver that supports the 'elasticnet' penalty
'penalty': ['elasticnet'], # use elasticnet penalty
'l1_ratio': [0.1, 0.5, 0.9], # controls the balance between L1 and L2 regularization
'max_iter': [2000, 5000, 10000], # increased the max_iter to allow more iterations for convergence
'C': [0.001, 0.01, 0.1, 1.0], # regularization strength; smaller values are stronger regularization
'tol': [1e-5, 1e-4, 1e-3], # tolerance for stopping criteria
}
```
### for the RF classifier (tuning_parameters_RF.txt)
https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html
```
{
'n_estimators': [50, 100, 200], # reasonable choices for number of trees
'max_depth': [10, 15, 20],  # limit max_depth
'min_samples_split': [2, 5, 10], # adjust this to avoid overfitting
'max_features': ['sqrt', 'log2'], # limit number of features used in each tree
'bootstrap': [True, False] # control bootstrapping
}
```
### for the SVC classifier (tuning_parameters_SVC.txt)
https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html
```
{
'kernel': ['linear', 'rbf'], # keeping only simpler kernels
'C': [0.1, 1.0, 10], # regularization parameter
'gamma': ['scale', 0.001], # kernel coefficient for 'rbf'
'max_iter': [1000, -1], # increase iterations
'tol': [1e-4, 1e-5], # lower tolerance for more strict convergence
}
```
### for the XGB classifier (tuning_parameters_XGB.txt)
https://xgboost.readthedocs.io/en/stable/python/python_api.html#xgboost.XGBClassifier
```
{
'max_depth': [3, 4, 5], # shallow trees for faster training
'eta': [0.1, 0.2, 0.3], # lower learning rate for more stable training
'max_leaves': [2, 4], # simpler trees for faster computation
'subsample': [0.8, 0.9], # subsample data to speed up training
'colsample_bytree': [0.7, 0.8], # feature subsampling to speed up training
'n_estimators': [50, 100], # fewer boosting rounds to speed up training
}
```
## genomic profils for prediction (e.g. genomic_profiles_for_prediction.tsv). "A" and "L" stand for alleles and locus, respectively.
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
git clone --branch v1.1.0 --single-branch https://github.com/Nicolas-Radomski/GenomicBasedClassification.git
cd GenomicBasedClassification
```
## with Python libraries
### for the DT classifier
```
python3.12 GenomicBasedClassification.py modeling -m genomic_profiles_for_modeling.tsv -ph phenotype_dataset.tsv -o MyDirectory -x DT_FirstAnalysis -da random -s 80 -c DT -k 5 -pa tuning_parameters_DT.txt -de 20
python3.12 GenomicBasedClassification.py prediction -m genomic_profiles_for_prediction.tsv -t MyDirectory/DT_FirstAnalysis_model.obj -f MyDirectory/DT_FirstAnalysis_features.obj -ef MyDirectory/DT_FirstAnalysis_encoded_features.obj -o MyDirectory -x DT_SecondAnalysis -de 20
```
### for the KNN classifier
```
python3.12 GenomicBasedClassification.py modeling -m genomic_profiles_for_modeling.tsv -ph MyDirectory/DT_FirstAnalysis_phenotype_dataset.tsv -o MyDirectory -x KNN_FirstAnalysis -da manual -c KNN -k 5 -pa tuning_parameters_KNN.txt -de 20
python3.12 GenomicBasedClassification.py prediction -m genomic_profiles_for_prediction.tsv -t MyDirectory/KNN_FirstAnalysis_model.obj -f MyDirectory/KNN_FirstAnalysis_features.obj -ef MyDirectory/KNN_FirstAnalysis_encoded_features.obj -o MyDirectory -x KNN_SecondAnalysis -de 20
```
### for the LR classifier
```
python3.12 GenomicBasedClassification.py modeling -m genomic_profiles_for_modeling.tsv -ph MyDirectory/DT_FirstAnalysis_phenotype_dataset.tsv -o MyDirectory -x LR_FirstAnalysis -da manual -c LR -k 5 -pa tuning_parameters_LR.txt -de 20
python3.12 GenomicBasedClassification.py prediction -m genomic_profiles_for_prediction.tsv -t MyDirectory/LR_FirstAnalysis_model.obj -f MyDirectory/LR_FirstAnalysis_features.obj -ef MyDirectory/LR_FirstAnalysis_encoded_features.obj -o MyDirectory -x LR_SecondAnalysis -de 20
```
### for the RF classifier
```
python3.12 GenomicBasedClassification.py modeling -m genomic_profiles_for_modeling.tsv -ph MyDirectory/DT_FirstAnalysis_phenotype_dataset.tsv -o MyDirectory -x RF_FirstAnalysis -da manual -c RF -k 5 -pa tuning_parameters_RF.txt -de 20
python3.12 GenomicBasedClassification.py prediction -m genomic_profiles_for_prediction.tsv -t MyDirectory/RF_FirstAnalysis_model.obj -f MyDirectory/RF_FirstAnalysis_features.obj -ef MyDirectory/RF_FirstAnalysis_encoded_features.obj -o MyDirectory -x RF_SecondAnalysis -de 20
```
### for the SVC classifier
```
python3.12 GenomicBasedClassification.py modeling -m genomic_profiles_for_modeling.tsv -ph MyDirectory/DT_FirstAnalysis_phenotype_dataset.tsv -o MyDirectory -x SVC_FirstAnalysis -da manual -c SVC -k 5 -pa tuning_parameters_SVC.txt -de 20
python3.12 GenomicBasedClassification.py prediction -m genomic_profiles_for_prediction.tsv -t MyDirectory/SVC_FirstAnalysis_model.obj -f MyDirectory/SVC_FirstAnalysis_features.obj -ef MyDirectory/SVC_FirstAnalysis_encoded_features.obj -o MyDirectory -x SVC_SecondAnalysis -de 20
```
### for the XGB classifier
```
python3.12 GenomicBasedClassification.py modeling -m genomic_profiles_for_modeling.tsv -ph MyDirectory/DT_FirstAnalysis_phenotype_dataset.tsv -o MyDirectory -x XGB_FirstAnalysis -da manual -c XGB -k 2 -pa tuning_parameters_XGB.txt -de 20
python3.12 GenomicBasedClassification.py prediction -m genomic_profiles_for_prediction.tsv -t MyDirectory/XGB_FirstAnalysis_model.obj -f MyDirectory/XGB_FirstAnalysis_features.obj -ef MyDirectory/XGB_FirstAnalysis_encoded_features.obj -ec MyDirectory/XGB_FirstAnalysis_encoded_classes.obj -o MyDirectory -x XGB_SecondAnalysis -de 20
```
## with a Docker image
### for the DT classifiers
```
docker run --rm --name nicolas -u $(id -u):$(id -g) -v $(pwd):/wd nicolasradomski/genomicbasedclassification:1.1.0 modeling -m genomic_profiles_for_modeling.tsv -ph phenotype_dataset.tsv -o MyDirectoryDockerHub -x DT_FirstAnalysis -da random -s 80 -c DT -k 5 -pa tuning_parameters_DT.txt -de 20
docker run --rm --name nicolas -u $(id -u):$(id -g) -v $(pwd):/wd nicolasradomski/genomicbasedclassification:1.1.0 prediction -m genomic_profiles_for_prediction.tsv -t MyDirectoryDockerHub/DT_FirstAnalysis_model.obj -f MyDirectoryDockerHub/DT_FirstAnalysis_features.obj -ef MyDirectoryDockerHub/DT_FirstAnalysis_encoded_features.obj -o MyDirectoryDockerHub -x DT_SecondAnalysis -de 20
```
### for the KNN classifier
```
docker run --rm --name nicolas -u $(id -u):$(id -g) -v $(pwd):/wd nicolasradomski/genomicbasedclassification:1.1.0 modeling -m genomic_profiles_for_modeling.tsv -ph MyDirectoryDockerHub/DT_FirstAnalysis_phenotype_dataset.tsv -o MyDirectoryDockerHub -x KNN_FirstAnalysis -da manual -c KNN -k 5 -pa tuning_parameters_KNN.txt -de 20
docker run --rm --name nicolas -u $(id -u):$(id -g) -v $(pwd):/wd nicolasradomski/genomicbasedclassification:1.1.0 prediction -m genomic_profiles_for_prediction.tsv -t MyDirectoryDockerHub/KNN_FirstAnalysis_model.obj -f MyDirectoryDockerHub/KNN_FirstAnalysis_features.obj -ef MyDirectoryDockerHub/KNN_FirstAnalysis_encoded_features.obj -o MyDirectoryDockerHub -x KNN_SecondAnalysis -de 20
```
### for the LR classifier
```
docker run --rm --name nicolas -u $(id -u):$(id -g) -v $(pwd):/wd nicolasradomski/genomicbasedclassification:1.1.0 modeling -m genomic_profiles_for_modeling.tsv -ph MyDirectoryDockerHub/DT_FirstAnalysis_phenotype_dataset.tsv -o MyDirectoryDockerHub -x LR_FirstAnalysis -da manual -c LR -k 5 -pa tuning_parameters_LR.txt -de 20
docker run --rm --name nicolas -u $(id -u):$(id -g) -v $(pwd):/wd nicolasradomski/genomicbasedclassification:1.1.0 prediction -m genomic_profiles_for_prediction.tsv -t MyDirectoryDockerHub/LR_FirstAnalysis_model.obj -f MyDirectoryDockerHub/LR_FirstAnalysis_features.obj -ef MyDirectoryDockerHub/LR_FirstAnalysis_encoded_features.obj -o MyDirectoryDockerHub -x LR_SecondAnalysis -de 20
```
### for the RF classifier
```
docker run --rm --name nicolas -u $(id -u):$(id -g) -v $(pwd):/wd nicolasradomski/genomicbasedclassification:1.1.0 modeling -m genomic_profiles_for_modeling.tsv -ph MyDirectoryDockerHub/DT_FirstAnalysis_phenotype_dataset.tsv -o MyDirectoryDockerHub -x RF_FirstAnalysis -da manual -c RF -k 5 -pa tuning_parameters_RF.txt -de 20
docker run --rm --name nicolas -u $(id -u):$(id -g) -v $(pwd):/wd nicolasradomski/genomicbasedclassification:1.1.0 prediction -m genomic_profiles_for_prediction.tsv -t MyDirectoryDockerHub/RF_FirstAnalysis_model.obj -f MyDirectoryDockerHub/RF_FirstAnalysis_features.obj -ef MyDirectoryDockerHub/RF_FirstAnalysis_encoded_features.obj -o MyDirectoryDockerHub -x RF_SecondAnalysis -de 20
```
### for the SVC classifier
```
docker run --rm --name nicolas -u $(id -u):$(id -g) -v $(pwd):/wd nicolasradomski/genomicbasedclassification:1.1.0 modeling -m genomic_profiles_for_modeling.tsv -ph MyDirectoryDockerHub/DT_FirstAnalysis_phenotype_dataset.tsv -o MyDirectoryDockerHub -x SVC_FirstAnalysis -da manual -c SVC -k 5 -pa tuning_parameters_SVC.txt -de 20
docker run --rm --name nicolas -u $(id -u):$(id -g) -v $(pwd):/wd nicolasradomski/genomicbasedclassification:1.1.0 prediction -m genomic_profiles_for_prediction.tsv -t MyDirectoryDockerHub/SVC_FirstAnalysis_model.obj -f MyDirectoryDockerHub/SVC_FirstAnalysis_features.obj -ef MyDirectoryDockerHub/SVC_FirstAnalysis_encoded_features.obj -o MyDirectoryDockerHub -x SVC_SecondAnalysis -de 20
```
### for the XGB classifier
```
docker run --rm --name nicolas -u $(id -u):$(id -g) -v $(pwd):/wd nicolasradomski/genomicbasedclassification:1.1.0 modeling -m genomic_profiles_for_modeling.tsv -ph MyDirectoryDockerHub/DT_FirstAnalysis_phenotype_dataset.tsv -o MyDirectoryDockerHub -x XGB_FirstAnalysis -da manual -c XGB -k 2 -pa tuning_parameters_XGB.txt -de 20
docker run --rm --name nicolas -u $(id -u):$(id -g) -v $(pwd):/wd nicolasradomski/genomicbasedclassification:1.1.0 prediction -m genomic_profiles_for_prediction.tsv -t MyDirectoryDockerHub/XGB_FirstAnalysis_model.obj -f MyDirectoryDockerHub/XGB_FirstAnalysis_features.obj -ef MyDirectoryDockerHub/XGB_FirstAnalysis_encoded_features.obj -ec MyDirectoryDockerHub/XGB_FirstAnalysis_encoded_classes.obj -o MyDirectoryDockerHub -x XGB_SecondAnalysis -de 20
```
## with a Conda environment
### for the DT classifiers
```
python GenomicBasedClassification.py modeling -m genomic_profiles_for_modeling.tsv -ph phenotype_dataset.tsv -o MyDirectoryConda -x DT_FirstAnalysis -da random -s 80 -c DT -k 5 -pa tuning_parameters_DT.txt -de 20
python GenomicBasedClassification.py prediction -m genomic_profiles_for_prediction.tsv -t MyDirectoryConda/DT_FirstAnalysis_model.obj -f MyDirectoryConda/DT_FirstAnalysis_features.obj -ef MyDirectoryConda/DT_FirstAnalysis_encoded_features.obj -o MyDirectoryConda -x DT_SecondAnalysis -de 20
```
### for the KNN classifier
```
python GenomicBasedClassification.py modeling -m genomic_profiles_for_modeling.tsv -ph MyDirectoryConda/DT_FirstAnalysis_phenotype_dataset.tsv -o MyDirectoryConda -x KNN_FirstAnalysis -da manual -c KNN -k 5 -pa tuning_parameters_KNN.txt -de 20
python GenomicBasedClassification.py prediction -m genomic_profiles_for_prediction.tsv -t MyDirectoryConda/KNN_FirstAnalysis_model.obj -f MyDirectoryConda/KNN_FirstAnalysis_features.obj -ef MyDirectoryConda/KNN_FirstAnalysis_encoded_features.obj -o MyDirectoryConda -x KNN_SecondAnalysis -de 20
```
### for the LR classifier
```
python GenomicBasedClassification.py modeling -m genomic_profiles_for_modeling.tsv -ph MyDirectoryConda/DT_FirstAnalysis_phenotype_dataset.tsv -o MyDirectoryConda -x LR_FirstAnalysis -da manual -c LR -k 5 -pa tuning_parameters_LR.txt -de 20
python GenomicBasedClassification.py prediction -m genomic_profiles_for_prediction.tsv -t MyDirectoryConda/LR_FirstAnalysis_model.obj -f MyDirectoryConda/LR_FirstAnalysis_features.obj -ef MyDirectoryConda/LR_FirstAnalysis_encoded_features.obj -o MyDirectoryConda -x LR_SecondAnalysis -de 20
```
### for the RF classifier
```
python GenomicBasedClassification.py modeling -m genomic_profiles_for_modeling.tsv -ph MyDirectoryConda/DT_FirstAnalysis_phenotype_dataset.tsv -o MyDirectoryConda -x RF_FirstAnalysis -da manual -c RF -k 5 -pa tuning_parameters_RF.txt -de 20
python GenomicBasedClassification.py prediction -m genomic_profiles_for_prediction.tsv -t MyDirectoryConda/RF_FirstAnalysis_model.obj -f MyDirectoryConda/RF_FirstAnalysis_features.obj -ef MyDirectoryConda/RF_FirstAnalysis_encoded_features.obj -o MyDirectoryConda -x RF_SecondAnalysis -de 20
```
### for the SVC classifier
```
python GenomicBasedClassification.py modeling -m genomic_profiles_for_modeling.tsv -ph MyDirectoryConda/DT_FirstAnalysis_phenotype_dataset.tsv -o MyDirectoryConda -x SVC_FirstAnalysis -da manual -c SVC -k 5 -pa tuning_parameters_SVC.txt -de 20
python GenomicBasedClassification.py prediction -m genomic_profiles_for_prediction.tsv -t MyDirectoryConda/SVC_FirstAnalysis_model.obj -f MyDirectoryConda/SVC_FirstAnalysis_features.obj -ef MyDirectoryConda/SVC_FirstAnalysis_encoded_features.obj -o MyDirectoryConda -x SVC_SecondAnalysis -de 20
```
### for the XGB classifier
```
python GenomicBasedClassification.py modeling -m genomic_profiles_for_modeling.tsv -ph MyDirectoryConda/DT_FirstAnalysis_phenotype_dataset.tsv -o MyDirectoryConda -x XGB_FirstAnalysis -da manual -c XGB -k 2 -pa tuning_parameters_XGB.txt -de 20
python GenomicBasedClassification.py prediction -m genomic_profiles_for_prediction.tsv -t MyDirectoryConda/XGB_FirstAnalysis_model.obj -f MyDirectoryConda/XGB_FirstAnalysis_features.obj -ef MyDirectoryConda/XGB_FirstAnalysis_encoded_features.obj -ec MyDirectoryConda/XGB_FirstAnalysis_encoded_classes.obj -o MyDirectoryConda -x XGB_SecondAnalysis -de 20
```
## with a Conda package
## for the DT classifiers
```
genomicbasedclassification modeling -m genomic_profiles_for_modeling.tsv -ph phenotype_dataset.tsv -o MyDirectoryAnaconda -x DT_FirstAnalysis -da random -s 80 -c DT -k 5 -pa tuning_parameters_DT.txt -de 20
genomicbasedclassification prediction -m genomic_profiles_for_prediction.tsv -t MyDirectoryAnaconda/DT_FirstAnalysis_model.obj -f MyDirectoryAnaconda/DT_FirstAnalysis_features.obj -ef MyDirectoryAnaconda/DT_FirstAnalysis_encoded_features.obj -o MyDirectoryAnaconda -x DT_SecondAnalysis -de 20
```
## for the KNN classifier
```
genomicbasedclassification modeling -m genomic_profiles_for_modeling.tsv -ph MyDirectoryAnaconda/DT_FirstAnalysis_phenotype_dataset.tsv -o MyDirectoryAnaconda -x KNN_FirstAnalysis -da manual -c KNN -k 5 -pa tuning_parameters_KNN.txt -de 20
genomicbasedclassification prediction -m genomic_profiles_for_prediction.tsv -t MyDirectoryAnaconda/KNN_FirstAnalysis_model.obj -f MyDirectoryAnaconda/KNN_FirstAnalysis_features.obj -ef MyDirectoryAnaconda/KNN_FirstAnalysis_encoded_features.obj -o MyDirectoryAnaconda -x KNN_SecondAnalysis -de 20
```
## for the LR classifier
```
genomicbasedclassification modeling -m genomic_profiles_for_modeling.tsv -ph MyDirectoryAnaconda/DT_FirstAnalysis_phenotype_dataset.tsv -o MyDirectoryAnaconda -x LR_FirstAnalysis -da manual -c LR -k 5 -pa tuning_parameters_LR.txt -de 20
genomicbasedclassification prediction -m genomic_profiles_for_prediction.tsv -t MyDirectoryAnaconda/LR_FirstAnalysis_model.obj -f MyDirectoryAnaconda/LR_FirstAnalysis_features.obj -ef MyDirectoryAnaconda/LR_FirstAnalysis_encoded_features.obj -o MyDirectoryAnaconda -x LR_SecondAnalysis -de 20
```
## for the RF classifier
```
genomicbasedclassification modeling -m genomic_profiles_for_modeling.tsv -ph MyDirectoryAnaconda/DT_FirstAnalysis_phenotype_dataset.tsv -o MyDirectoryAnaconda -x RF_FirstAnalysis -da manual -c RF -k 5 -pa tuning_parameters_RF.txt -de 20
genomicbasedclassification prediction -m genomic_profiles_for_prediction.tsv -t MyDirectoryAnaconda/RF_FirstAnalysis_model.obj -f MyDirectoryAnaconda/RF_FirstAnalysis_features.obj -ef MyDirectoryAnaconda/RF_FirstAnalysis_encoded_features.obj -o MyDirectoryAnaconda -x RF_SecondAnalysis -de 20
```
## for the SVC classifier
```
genomicbasedclassification modeling -m genomic_profiles_for_modeling.tsv -ph MyDirectoryAnaconda/DT_FirstAnalysis_phenotype_dataset.tsv -o MyDirectoryAnaconda -x SVC_FirstAnalysis -da manual -c SVC -k 5 -pa tuning_parameters_SVC.txt -de 20
genomicbasedclassification prediction -m genomic_profiles_for_prediction.tsv -t MyDirectoryAnaconda/SVC_FirstAnalysis_model.obj -f MyDirectoryAnaconda/SVC_FirstAnalysis_features.obj -ef MyDirectoryAnaconda/SVC_FirstAnalysis_encoded_features.obj -o MyDirectoryAnaconda -x SVC_SecondAnalysis -de 20
```
## for the XGB classifier
```
genomicbasedclassification modeling -m genomic_profiles_for_modeling.tsv -ph MyDirectoryAnaconda/DT_FirstAnalysis_phenotype_dataset.tsv -o MyDirectoryAnaconda -x XGB_FirstAnalysis -da manual -c XGB -k 2 -pa tuning_parameters_XGB.txt -de 20
genomicbasedclassification prediction -m genomic_profiles_for_prediction.tsv -t MyDirectoryAnaconda/XGB_FirstAnalysis_model.obj -f MyDirectoryAnaconda/XGB_FirstAnalysis_features.obj -ef MyDirectoryAnaconda/XGB_FirstAnalysis_encoded_features.obj -ec MyDirectoryAnaconda/XGB_FirstAnalysis_encoded_classes.obj -o MyDirectoryAnaconda -x XGB_SecondAnalysis -de 20
```
# Examples of expected output (see inclosed directory called 'MyDirectory')
## confusion matrix
```
from the training dataset: 
phenotype  fruit  pig  poultry
    fruit     49    3        4
      pig      0   56        0
  poultry      0    5       43
from the testing dataset: 
phenotype  fruit  pig  poultry
    fruit     11    2        1
      pig      0   14        0
  poultry      0    0       12
NB: The expectation and prediction are represented by rows and columns, respectively. 
```
## metrics per class
```
from the training dataset: 
phenotype  TN  FP  FN  TP  support  accuracy  sensitivity  specificity  precision   recall  f1-score      MCC
    fruit 104   0   7  49       56   0.95625     0.875000     1.000000   1.000000 0.875000  0.933333 0.905439
      pig  96   8   0  56       56   0.95000     1.000000     0.923077   0.875000 1.000000  0.933333 0.898717
  poultry 108   4   5  43       48   0.94375     0.895833     0.964286   0.914894 0.895833  0.905263 0.865366
from the testing dataset: 
phenotype  TN  FP  FN  TP  support  accuracy  sensitivity  specificity  precision   recall  f1-score      MCC
    fruit  26   0   3  11       14     0.925     0.785714     1.000000   1.000000 0.785714  0.880000 0.839305
      pig  24   2   0  14       14     0.950     1.000000     0.923077   0.875000 1.000000  0.933333 0.898717
  poultry  27   1   0  12       12     0.975     1.000000     0.964286   0.923077 1.000000  0.960000 0.943456
NB: The term 'support' corresponds to TP + FN. 
NB: MCC stands for Matthews Correlation Coefficient. 
NB: Sensitivity and recall must be equal, as they are based on the same formula. 
```
## global metrics
```
from the training dataset: 
 accuracy  sensitivity  specificity  precision   recall  f1-score      MCC  Cohen kappa
    0.925     0.923611     0.962454   0.929965 0.923611  0.923977 0.890153     0.887165
from the testing dataset: 
 accuracy  sensitivity  specificity  precision   recall  f1-score      MCC  Cohen kappa
    0.925     0.928571     0.962454   0.932692 0.928571  0.924444 0.893306      0.88743
NB: MCC stands for Matthews Correlation Coefficient. 
NB: Sensitivity and recall must be equal, as they are based on the same formula. 
```
## prediction
```
 sample prediction    fruit      pig  poultry
S2.1.01        pig 0.034727 0.905110 0.060162
S2.1.02        pig 0.047815 0.869349 0.082836
S2.1.03    poultry 0.016830 0.005268 0.977902
S2.1.04        pig 0.121309 0.787437 0.091254
S2.1.05      fruit 0.523366 0.069992 0.406642
S2.1.06    poultry 0.047640 0.014913 0.937446
S2.1.07        pig 0.243543 0.611796 0.144661
S2.1.08    poultry 0.011523 0.003607 0.984870
S2.1.09        pig 0.048634 0.932713 0.018653
S2.1.10        pig 0.153262 0.778802 0.067936
```
# Illustration
![workflow figure](https://github.com/Nicolas-Radomski/GenomicBasedClassification/blob/main/illustration.png)
# Funding
Ricerca Corrente - IZS AM 06/24 RC: "genomic data-based machine learning to predict categorical and continuous phenotypes by classification and regression".
# Acknowledgements
Many thanks to Andrea De Ruvo and Adriano Di Pasquale for the insightful discussions that helped improve the algorithm.
# Reference
Pierluigi Castelli, Andrea De Ruvo, Andrea Bucciacchio, Nicola D'Alterio, Cesare Camma, Adriano Di Pasquale and Nicolas Radomski (2023) Harmonization of supervised machine learning practices for efficient source attribution of Listeria monocytogenes based on genomic data. 2023, BMC Genomics, 24(560):1-19, https://doi.org/10.1186/s12864-023-09667-w
# Repositories
- GitHub: https://github.com/Nicolas-Radomski/GenomicBasedClassification
- Docker Hub: https://hub.docker.com/r/nicolasradomski/genomicbasedclassification
- Anaconda Hub: https://anaconda.org/nicolasradomski/genomicbasedclassification
- R users: https://github.com/Nicolas-Radomski/GenomicBasedMachineLearning
# Author
Nicolas Radomski
