import argparse
from pathlib import Path

from src.data import (
    get_train_test_data,
    parse_features,
)
from src.fairness import (
    run_fairness,
    save_fairness_outputs,
)
from src.plots import save_all_plots
from src.train import (
    evaluate_model,
    save_cv_results,
    save_metrics,
    save_model,
)


def make_parser():
    parser = argparse.ArgumentParser()

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
        help="Comma-separated features to exclude from the chosen feature set.",
    )

    parser.add_argument(
        "--data-path",
        type=str,
        default=None,
    )

    parser.add_argument(
        "--cv",
        type=int,
        default=5,
    )

    return parser


def run_model_script(model_name, fit_function):
    parser = make_parser()
    args = parser.parse_args()

    custom_features = parse_features(args.features)

    exclude_features = parse_features(args.exclude_features)

    X_train, X_test, y_train, y_test, feature_names = get_train_test_data(
        feature_set=args.feature_set,
        custom_features=custom_features,
        exclude_features=exclude_features,
        data_path=args.data_path,
    )

    print("\nUsing features:")
    print(feature_names)

    model, search = fit_function(
        X_train,
        y_train,
        feature_names,
        cv=args.cv,
    )

    prefix = f"{model_name}_{args.feature_set}"

    if exclude_features:
        prefix += "_excluded_" + "_".join(exclude_features)

    metrics, report, cm = evaluate_model(
        model,
        X_train,
        y_train,
        X_test,
        y_test,
    )

    print("\nBest parameters:")
    print(search.best_params_)

    print("\nMetrics:")
    print(metrics)

    print("\nClassification report:")
    print(report)

    print("\nConfusion matrix:")
    print(cm)

    save_model(
        model,
        f"models/{prefix}.joblib",
    )

    save_cv_results(
        search,
        f"results/tables/{prefix}_cv_results.csv",
    )

    save_metrics(
        metrics,
        f"results/tables/{prefix}_metrics.csv",
    )

    Path("results/tables").mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        f"results/tables/{prefix}_classification_report.txt",
        "w",
    ) as f:
        f.write(report)

    fairness_results = run_fairness(
        model,
        X_test,
        y_test,
    )

    save_fairness_outputs(
        fairness_results,
        prefix,
    )

    save_all_plots(
        model,
        X_test,
        y_test,
        fairness_results,
        prefix,
    )

    print(f"\nSaved outputs using prefix: {prefix}")