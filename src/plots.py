from pathlib import Path

import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay, RocCurveDisplay


def save_all_plots(model, X_test, y_test, fairness_results, prefix):
    Path("results/figures").mkdir(parents=True, exist_ok=True)

    save_roc_curve(
        model,
        X_test,
        y_test,
        f"results/figures/{prefix}_roc_curve.png",
    )

    save_confusion_matrix(
        model,
        X_test,
        y_test,
        f"results/figures/{prefix}_confusion_matrix.png",
    )

    save_combined_fairness_plot(
        fairness_results,
        prefix,
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


def save_combined_fairness_plot(fairness_results, prefix):
    metric_tables = fairness_results["metric_tables"]

    group_order = [
        ("GenderGroup", "By Gender"),
        ("EthnicityGroup", "By Ethnicity"),
        ("EducationGroup", "By Education"),
        ("AgeGroup", "By Age Group"),
    ]

    metrics = [
        ("accuracy", "Accuracy"),
        ("TPR", "Sensitivity (TPR)"),
        ("FPR", "False Positive Rate"),
        ("F1", "F1"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    axes = axes.flatten()

    for ax, (group_col, title) in zip(axes, group_order):

        table = metric_tables[group_col]

        if table.empty:
            ax.set_visible(False)
            continue

        x = range(len(table))

        width = 0.2

        for i, (metric_col, label) in enumerate(metrics):
            ax.bar(
                [v + i * width for v in x],
                table[metric_col],
                width=width,
                label=label,
            )

        ax.set_xticks([v + 1.5 * width for v in x])

        ax.set_xticklabels(
            table["group"].astype(str),
            rotation=20,
            ha="right",
        )

        ax.set_ylim(0, 1.15)

        ax.set_title(title)

        ax.set_ylabel("Score")

        ax.grid(axis="y", alpha=0.3)

    handles, labels = axes[0].get_legend_handles_labels()

    fig.legend(
        handles,
        labels,
        loc="upper right",
    )

    fig.suptitle(
        f"{prefix.replace('_', ' ').title()} — Demographic Subgroup Performance",
        fontsize=16,
    )

    plt.tight_layout()

    plt.savefig(
        f"results/figures/{prefix}_combined_fairness.png",
        bbox_inches="tight",
    )

    plt.close()