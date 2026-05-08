# Alzheimer's ML Project

File structure:

- `src/config.py`
  - constants, feature lists, random seed, demographic label mappings

- `src/data.py`
  - load dataset
  - choose feature subsets
  - exclude selected features
  - split train/test data
  - feature grouping utilities

- `src/train.py`
  - preprocessing pipelines
  - grid-search cross-validation
  - model training
  - model evaluation
  - save trained models/results

- `src/fairness.py`
  - subgroup fairness metrics
  - demographic parity tests
  - equal opportunity tests
  - equalized odds tests

- `src/plots.py`
  - ROC curves
  - confusion matrices
  - combined demographic fairness plots

- `scripts/train_logistic.py`
  - runs the full logistic regression experiment

- `scripts/train_svm.py`
  - runs the full SVM experiment

- `scripts/train_boosted_trees.py`
  - runs the full boosted-tree experiment

- `scripts/run_clustering.py`
  - runs clustering experiments and clustering metrics

- `scripts/find_significant_fairness_tests.py`
  - finds statistically significant fairness test results across saved CSVs


## Installation

Install dependencies:

 
pip install -r requirements.txt




## Running the Supervised Models

From the project root:


python -m scripts.train_logistic
python -m scripts.train_svm
python -m scripts.train_boosted_trees


Each script:

1. loads the dataset
2. preprocesses the features
3. performs grid-search cross-validation
4. trains the model
5. evaluates performance
6. computes fairness metrics/tests
7. saves plots, tables, and trained models



## Feature Sets

### Selected Features


--feature-set selected


Uses a curated subset of demographic, medical, and clinical variables.

Example:


python -m scripts.train_svm --feature-set selected




### All Features


--feature-set all


Uses all columns except:


PatientID
DoctorInCharge
Diagnosis


Example:


python -m scripts.train_svm --feature-set all




### Custom Features


--feature-set custom --features Age,BMI,MMSE,ADL


Uses only the comma-separated features provided.

Example:


python -m scripts.train_logistic --feature-set custom --features Age,BMI,MMSE,ADL




## Excluding Features

You can remove specific variables from any feature set using:


--exclude-features


Example:


python -m scripts.train_svm \
    --feature-set all \
    --exclude-features MMSE,ADL,FunctionalAssessment


Another example:


python -m scripts.train_logistic \
    --feature-set selected \
    --exclude-features Age,BMI


This is useful for ablation studies and testing how model performance changes when specific variables are removed.



## Supervised Models

The project trains:

- Logistic Regression
- Support Vector Machine
- Boosted Decision Trees (AdaBoost)

using grid-search cross-validation.

Continuous variables are standardized before training.

Categorical variables are one-hot encoded.



## Performance Metrics

The supervised experiments report:

- train accuracy
- test accuracy
- precision
- recall
- F1
- ROC-AUC
- confusion matrix
- classification report



## Fairness / Subgroup Analysis

Fairness analysis is performed across:

- gender
- ethnicity
- education level
- age groups

For each subgroup, the project computes:

- subgroup size
- prevalence
- predicted positive rate
- accuracy
- TPR / sensitivity
- FPR
- PPV
- NPV
- F1
- ROC-AUC

The project also performs statistical fairness tests:

- demographic parity chi-square tests
- error-rate parity chi-square tests
- pairwise TPR z-tests, for equal opportunity
- pairwise FPR z-tests, for equalized odds

A small p-value indicates evidence of subgroup disparity, not definitive proof that a model is unfair.



## Fairness Plots

Each model produces a combined subgroup-performance figure containing:

- accuracy
- sensitivity / TPR
- false positive rate
- F1

across demographic groups.



## Clustering

Run clustering experiments:


python -m scripts.run_clustering


Run with all features:


python -m scripts.run_clustering --feature-set all


The clustering script runs:

- KMeans
- Agglomerative Clustering
- Gaussian Mixture Models

and reports:

- Adjusted Rand Index (ARI)
- Normalized Mutual Information (NMI)

The diagnosis label is NOT used during clustering.
It is only used afterward for evaluation.



## Finding Significant Fairness Tests

After running one or more supervised experiments:


python -m scripts.find_significant_fairness_tests


Use a stricter significance threshold:


python -m scripts.find_significant_fairness_tests --alpha 0.01


This scans the saved fairness CSVs and outputs statistically significant results.



## Outputs

Generated outputs are saved to:


results/tables/
results/figures/
models/


These generated files are ignored by Git except for placeholder `.gitkeep` files.