
from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.ensemble import AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from src.config import RANDOM_STATE
from src.data import feature_groups


def make_preprocessor(feature_names):
    continuous, categorical, binary = feature_groups(feature_names)

    return ColumnTransformer(
        transformers=[
            ("continuous", StandardScaler(), continuous),
            (
                "categorical",
                OneHotEncoder(drop="first", handle_unknown="ignore"),
                categorical,
            ),
            ("binary", "passthrough", binary),
        ],
        remainder="drop",
    )


def fit_logistic_regression(X_train, y_train, feature_names, cv=5):
    model = Pipeline(
        steps=[
            ("preprocessor", make_preprocessor(feature_names)),
            (
                "classifier",
                LogisticRegression(
                    max_iter=5000,
                    class_weight="balanced",
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )

    grid = {
        "classifier__solver": ["liblinear"],
        "classifier__penalty": ["l1", "l2"],
        "classifier__C": [0.01, 0.1, 1, 10],
    }

    return _fit_grid(model, grid, X_train, y_train, cv=cv)


def fit_svm(X_train, y_train, feature_names, cv=5):
    model = Pipeline(
        steps=[
            ("preprocessor", make_preprocessor(feature_names)),
            (
                "classifier",
                SVC(
                    probability=True,
                    class_weight="balanced",
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )

    grid = {
        "classifier__C": [0.1, 1, 10],
        "classifier__kernel": ["linear", "rbf", "poly"],
        "classifier__gamma": ["scale", "auto"],
    }

    return _fit_grid(model, grid, X_train, y_train, cv=cv)


def fit_boosted_trees(X_train, y_train, feature_names, cv=5):
    max_components = min(len(feature_names), len(X_train) - 1)
    pca_candidates = [n for n in [10, 15, 20] if n <= max_components]

    if not pca_candidates:
        pca_candidates = [min(5, max_components)]

    model = Pipeline(
        steps=[
            ("preprocessor", make_preprocessor(feature_names)),
            ("pca", PCA()),
            (
                "classifier",
                AdaBoostClassifier(random_state=RANDOM_STATE),
            ),
        ]
    )

    grid = {
        "pca__n_components": pca_candidates,
        "classifier__estimator": [
            DecisionTreeClassifier(max_depth=2, random_state=RANDOM_STATE),
            DecisionTreeClassifier(max_depth=3, random_state=RANDOM_STATE),
        ],
        "classifier__n_estimators": [50, 100, 200],
        "classifier__learning_rate": [0.01, 0.1, 1.0],
    }

    return _fit_grid(model, grid, X_train, y_train, cv=cv)


def _fit_grid(model, grid, X_train, y_train, cv=5):
    search = GridSearchCV(
        model,
        grid,
        scoring="roc_auc",
        cv=cv,
        n_jobs=-1,
        refit=True,
        return_train_score=True,
    )

    search.fit(X_train, y_train)

    return search.best_estimator_, search


def get_scores(model, X):
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]

    return model.decision_function(X)


def evaluate_model(model, X_train, y_train, X_test, y_test):
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    y_test_score = get_scores(model, X_test)

    metrics = {
        "train_accuracy": accuracy_score(y_train, y_train_pred),
        "test_accuracy": accuracy_score(y_test, y_test_pred),
        "precision": precision_score(y_test, y_test_pred, zero_division=0),
        "recall": recall_score(y_test, y_test_pred, zero_division=0),
        "f1": f1_score(y_test, y_test_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_test, y_test_score),
    }

    report = classification_report(y_test, y_test_pred, zero_division=0)
    cm = confusion_matrix(y_test, y_test_pred)

    return metrics, report, cm


def save_model(model, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)


def save_cv_results(search, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(search.cv_results_).to_csv(path, index=False)


def save_metrics(metrics, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([metrics]).to_csv(path, index=False)
