
# Alzheimer's ML Project

Simpler structure:

- `src/data.py`: load data, choose features, split data
- `src/train.py`: train models and evaluate performance
- `src/fairness.py`: fairness metrics and tests
- `src/plots.py`: ROC, confusion matrix, and fairness plots
- `scripts/train_logistic.py`: logistic regression experiment
- `scripts/train_svm.py`: SVM experiment
- `scripts/train_boosted_trees.py`: boosted trees experiment

## Install

```bash
pip install -r requirements.txt
```

## Run

From the project root:

```bash
python -m scripts.train_logistic
python -m scripts.train_svm
python -m scripts.train_boosted_trees
```

Run with all features:

```bash
python -m scripts.train_logistic --feature-set all
```

Run with custom features:

```bash
python -m scripts.train_logistic --feature-set custom --features Age,BMI,MMSE,ADL
```

Outputs go to:

```text
results/tables/
results/figures/
models/
```
