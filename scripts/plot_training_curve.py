import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import AdaBoostClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils.class_weight import compute_class_weight

from src.config import RANDOM_STATE
from src.data import get_train_test_data, parse_features
from src.train import make_preprocessor


def get_scores(model, X):
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    return model.decision_function(X)


def compute_metrics(model, X_train, y_train, X_test, y_test):
    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)

    train_score = get_scores(model, X_train)
    test_score = get_scores(model, X_test)

    return {
        "train_accuracy": accuracy_score(y_train, train_pred),
        "test_accuracy": accuracy_score(y_test, test_pred),
        "train_roc_auc": roc_auc_score(y_train, train_score),
        "test_roc_auc": roc_auc_score(y_test, test_score),
    }


def plot_history(history, output_path, title):
    plt.figure(figsize=(9, 6))

    plt.plot(history["step"], history["train_accuracy"], label="Train Accuracy")
    plt.plot(history["step"], history["test_accuracy"], label="Test Accuracy")
    plt.plot(history["step"], history["train_roc_auc"], label="Train ROC-AUC")
    plt.plot(history["step"], history["test_roc_auc"], label="Test ROC-AUC")

    plt.xlabel("Training Step")
    plt.ylabel("Score")
    plt.ylim(0, 1.05)
    plt.title(title)
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()

    plt.savefig(output_path, bbox_inches="tight")
    plt.close()


def train_sgd_logistic_curve(
    X_train,
    y_train,
    X_test,
    y_test,
    feature_names,
    params,
    steps,
):
    preprocessor = make_preprocessor(feature_names)

    X_train_proc = preprocessor.fit_transform(X_train)
    X_test_proc = preprocessor.transform(X_test)


    classes = np.unique(y_train)

    class_weights_array = compute_class_weight(
        class_weight="balanced",
        classes=classes,
        y=y_train,
    )

    class_weights = dict(zip(classes, class_weights_array))

    model = SGDClassifier(
        loss="log_loss",
        penalty=params.get("penalty", "l2"),
        alpha=params.get("alpha", 0.0001),
        class_weight=class_weights,
        random_state=RANDOM_STATE,
        max_iter=1,
        tol=None,
        warm_start=True,
    )

    rows = []

    for step in range(1, steps + 1):
        model.partial_fit(
            X_train_proc,
            y_train,
            classes=classes,
        )

        rows.append(
            {
                "step": step,
                **compute_metrics(
                    model,
                    X_train_proc,
                    y_train,
                    X_test_proc,
                    y_test,
                ),
            }
        )

    return pd.DataFrame(rows)


def train_boosted_trees_curve(
    X_train,
    y_train,
    X_test,
    y_test,
    feature_names,
    params,
    steps,
):
    preprocessor = make_preprocessor(feature_names)

    X_train_proc = preprocessor.fit_transform(X_train)
    X_test_proc = preprocessor.transform(X_test)

    max_depth = params.get("max_depth", 2)
    learning_rate = params.get("learning_rate", 0.1)

    model = AdaBoostClassifier(
        estimator=DecisionTreeClassifier(
            max_depth=max_depth,
            random_state=RANDOM_STATE,
        ),
        n_estimators=steps,
        learning_rate=learning_rate,
        random_state=RANDOM_STATE,
    )

    model.fit(X_train_proc, y_train)

    rows = []

    for step, (
        train_pred,
        test_pred,
        train_score,
        test_score,
    ) in enumerate(
        zip(
            model.staged_predict(X_train_proc),
            model.staged_predict(X_test_proc),
            model.staged_predict_proba(X_train_proc),
            model.staged_predict_proba(X_test_proc),
        ),
        start=1,
    ):
        rows.append(
            {
                "step": step,
                "train_accuracy": accuracy_score(y_train, train_pred),
                "test_accuracy": accuracy_score(y_test, test_pred),
                "train_roc_auc": roc_auc_score(y_train, train_score[:, 1]),
                "test_roc_auc": roc_auc_score(y_test, test_score[:, 1]),
            }
        )

    return pd.DataFrame(rows)


def train_svm_curve(
    X_train,
    y_train,
    X_test,
    y_test,
    feature_names,
    params,
    steps,
):
    preprocessor = make_preprocessor(feature_names)

    X_train_proc = preprocessor.fit_transform(X_train)
    X_test_proc = preprocessor.transform(X_test)

    max_iters = list(range(100, 100 * steps + 1, 100))

    rows = []

    for i, max_iter in enumerate(max_iters, start=1):
        model = SVC(
            C=params.get("C", 1),
            kernel=params.get("kernel", "rbf"),
            gamma=params.get("gamma", "scale"),
            probability=True,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            max_iter=max_iter,
        )

        model.fit(X_train_proc, y_train)

        rows.append(
            {
                "step": max_iter,
                **compute_metrics(
                    model,
                    X_train_proc,
                    y_train,
                    X_test_proc,
                    y_test,
                ),
            }
        )

    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model",
        choices=["logistic", "svm", "boosted_trees"],
        required=True,
    )

    parser.add_argument(
        "--feature-set",
        choices=["selected", "all", "custom"],
        default="selected",
    )

    parser.add_argument(
        "--features",
        type=str,
        default=None,
    )

    parser.add_argument(
        "--exclude-features",
        type=str,
        default=None,
    )

    parser.add_argument(
        "--params",
        type=str,
        default="{}",
        help='JSON string of hyperparameters, e.g. \'{"alpha":0.0001,"penalty":"l2"}\'',
    )

    parser.add_argument(
        "--steps",
        type=int,
        default=100,
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/training_plots",
    )

    parser.add_argument(
        "--data-path",
        type=str,
        default=None,
    )

    args = parser.parse_args()

    params = json.loads(args.params)

    custom_features = parse_features(args.features)
    exclude_features = parse_features(args.exclude_features)

    X_train, X_test, y_train, y_test, feature_names = get_train_test_data(
        feature_set=args.feature_set,
        custom_features=custom_features,
        exclude_features=exclude_features,
        data_path=args.data_path,
    )

    if args.model == "logistic":
        history = train_sgd_logistic_curve(
            X_train,
            y_train,
            X_test,
            y_test,
            feature_names,
            params,
            args.steps,
        )

    elif args.model == "boosted_trees":
        history = train_boosted_trees_curve(
            X_train,
            y_train,
            X_test,
            y_test,
            feature_names,
            params,
            args.steps,
        )

    elif args.model == "svm":
        history = train_svm_curve(
            X_train,
            y_train,
            X_test,
            y_test,
            feature_names,
            params,
            args.steps,
        )

    else:
        raise ValueError(f"Unknown model: {args.model}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    params_name = "_".join(
        f"{key}-{value}" for key, value in params.items()
    )

    if not params_name:
        params_name = "default"

    prefix = f"{args.model}_{args.feature_set}_{params_name}"

    csv_path = output_dir / f"{prefix}_training_curve.csv"
    png_path = output_dir / f"{prefix}_training_curve.png"

    history.to_csv(csv_path, index=False)

    plot_history(
        history,
        png_path,
        title=f"{args.model} training curve: {params}",
    )

    print(f"Saved CSV to {csv_path}")
    print(f"Saved plot to {png_path}")


if __name__ == "__main__":
    main()