
from pathlib import Path

import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay, RocCurveDisplay


def save_all_plots(model, X_test, y_test, fairness_results, prefix):
    Path("results/figures").mkdir(parents=True, exist_ok=True)

    save_roc_curve(model, X_test, y_test, f"results/figures/{prefix}_roc_curve.png")
    save_confusion_matrix(model, X_test, y_test, f"results/figures/{prefix}_confusion_matrix.png")

    for group_col, table in fairness_results["metric_tables"].items():
        if table.empty:
            continue

        save_group_metric_bar(
            table,
            group_col,
            "TPR",
            f"results/figures/{prefix}_{group_col}_TPR.png",
        )

        save_group_metric_bar(
            table,
            group_col,
            "FPR",
            f"results/figures/{prefix}_{group_col}_FPR.png",
        )

        save_group_metric_bar(
            table,
            group_col,
            "accuracy",
            f"results/figures/{prefix}_{group_col}_accuracy.png",
        )


def save_roc_curve(model, X_test, y_test, output_path):
    RocCurveDisplay.from_estimator(model, X_test, y_test)
    plt.title("ROC Curve")
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()


def save_confusion_matrix(model, X_test, y_test, output_path):
    ConfusionMatrixDisplay.from_estimator(model, X_test, y_test)
    plt.title("Confusion Matrix")
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()


def save_group_metric_bar(table, group_col, metric, output_path):
    plot_df = table.copy()
    plot_df["group"] = plot_df["group"].astype(str)

    plt.figure(figsize=(8, 5))
    plt.bar(plot_df["group"], plot_df[metric])
    plt.xlabel(group_col)
    plt.ylabel(metric)
    plt.title(f"{metric} by {group_col}")
    plt.xticks(rotation=30, ha="right")
    plt.ylim(0, 1)
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()
