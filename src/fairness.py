
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import confusion_matrix, f1_score, roc_auc_score

from src.config import EDUCATION_MAP, ETHNICITY_MAP, GENDER_MAP
from src.train import get_scores


def make_fairness_dataframe(model, X_test, y_test):
    needed = ["Gender", "Ethnicity", "EducationLevel", "Age"]
    missing = [col for col in needed if col not in X_test.columns]

    if missing:
        raise ValueError(
            "Fairness requires Gender, Ethnicity, EducationLevel, and Age. "
            f"Missing from current feature set: {missing}"
        )

    df = X_test[needed].copy().reset_index(drop=True)
    df["y_true"] = np.asarray(y_test)
    df["y_pred"] = model.predict(X_test)
    df["y_score"] = get_scores(model, X_test)
    df["error"] = (df["y_true"] != df["y_pred"]).astype(int)

    df["GenderGroup"] = df["Gender"].map(GENDER_MAP)
    df["EthnicityGroup"] = df["Ethnicity"].map(ETHNICITY_MAP)
    df["EducationGroup"] = df["EducationLevel"].map(EDUCATION_MAP)
    df["AgeGroup"] = pd.cut(
        df["Age"],
        bins=[59, 69, 79, 90],
        labels=["60-69", "70-79", "80-90"],
    )

    return df


def subgroup_metrics(df, group_col):
    rows = []

    for group, sub in df.groupby(group_col, observed=False):
        if len(sub) < 5 or sub["y_true"].nunique() < 2:
            continue

        tn, fp, fn, tp = confusion_matrix(
            sub["y_true"],
            sub["y_pred"],
            labels=[0, 1],
        ).ravel()

        rows.append(
            {
                "group": group,
                "n": len(sub),
                "prevalence": sub["y_true"].mean(),
                "predicted_positive_rate": sub["y_pred"].mean(),
                "accuracy": (sub["y_true"] == sub["y_pred"]).mean(),
                "TPR": tp / (tp + fn) if tp + fn > 0 else np.nan,
                "FPR": fp / (fp + tn) if fp + tn > 0 else np.nan,
                "PPV": tp / (tp + fp) if tp + fp > 0 else np.nan,
                "NPV": tn / (tn + fn) if tn + fn > 0 else np.nan,
                "F1": f1_score(sub["y_true"], sub["y_pred"], zero_division=0),
                "ROC_AUC": roc_auc_score(sub["y_true"], sub["y_score"]),
            }
        )

    return pd.DataFrame(rows)


def chi_square(df, group_col, outcome_col, name):
    table = pd.crosstab(df[group_col], df[outcome_col])

    if table.shape[0] < 2 or table.shape[1] < 2:
        return {
            "test": name,
            "group_column": group_col,
            "chi2": np.nan,
            "p_value": np.nan,
            "dof": np.nan,
        }

    chi2, p, dof, _ = stats.chi2_contingency(table)

    return {
        "test": name,
        "group_column": group_col,
        "chi2": chi2,
        "p_value": p,
        "dof": dof,
    }


def z_test_rates(p1, n1, p2, n2):
    if n1 == 0 or n2 == 0:
        return np.nan, np.nan

    pooled = (p1 * n1 + p2 * n2) / (n1 + n2)
    se = np.sqrt(pooled * (1 - pooled) * (1 / n1 + 1 / n2))

    if se == 0:
        return np.nan, np.nan

    z = (p1 - p2) / se
    p = 2 * (1 - stats.norm.cdf(abs(z)))

    return z, p


def pairwise_tpr_fpr_tests(df, group_col):
    rows = []
    groups = [g for g in df[group_col].dropna().unique()]

    for g1, g2 in combinations(groups, 2):
        s1 = df[df[group_col] == g1]
        s2 = df[df[group_col] == g2]

        positives1 = s1[s1["y_true"] == 1]
        positives2 = s2[s2["y_true"] == 1]

        if len(positives1) > 0 and len(positives2) > 0:
            tpr1 = positives1["y_pred"].mean()
            tpr2 = positives2["y_pred"].mean()
            z, p = z_test_rates(tpr1, len(positives1), tpr2, len(positives2))

            rows.append(
                {
                    "test": "equal_opportunity_TPR",
                    "group_column": group_col,
                    "group_1": g1,
                    "group_2": g2,
                    "rate_1": tpr1,
                    "rate_2": tpr2,
                    "difference": tpr1 - tpr2,
                    "z": z,
                    "p_value": p,
                }
            )

        negatives1 = s1[s1["y_true"] == 0]
        negatives2 = s2[s2["y_true"] == 0]

        if len(negatives1) > 0 and len(negatives2) > 0:
            fpr1 = negatives1["y_pred"].mean()
            fpr2 = negatives2["y_pred"].mean()
            z, p = z_test_rates(fpr1, len(negatives1), fpr2, len(negatives2))

            rows.append(
                {
                    "test": "equalized_odds_FPR",
                    "group_column": group_col,
                    "group_1": g1,
                    "group_2": g2,
                    "rate_1": fpr1,
                    "rate_2": fpr2,
                    "difference": fpr1 - fpr2,
                    "z": z,
                    "p_value": p,
                }
            )

    return rows


def run_fairness(model, X_test, y_test):
    df = make_fairness_dataframe(model, X_test, y_test)

    group_cols = [
        "GenderGroup",
        "EthnicityGroup",
        "EducationGroup",
        "AgeGroup",
    ]

    metric_tables = {}
    omnibus_tests = []
    pairwise_tests = []

    for group_col in group_cols:
        metric_tables[group_col] = subgroup_metrics(df, group_col)

        omnibus_tests.append(
            chi_square(df, group_col, "y_pred", "demographic_parity_chi_square")
        )
        omnibus_tests.append(
            chi_square(df, group_col, "error", "error_rate_parity_chi_square")
        )

        pairwise_tests.extend(pairwise_tpr_fpr_tests(df, group_col))

    return {
        "fairness_data": df,
        "metric_tables": metric_tables,
        "omnibus_tests": pd.DataFrame(omnibus_tests),
        "pairwise_tests": pd.DataFrame(pairwise_tests),
    }


def save_fairness_outputs(fairness_results, prefix):
    Path("results/tables").mkdir(parents=True, exist_ok=True)

    fairness_results["fairness_data"].to_csv(
        f"results/tables/{prefix}_fairness_data.csv",
        index=False,
    )

    for group_col, table in fairness_results["metric_tables"].items():
        table.to_csv(
            f"results/tables/{prefix}_fairness_{group_col}.csv",
            index=False,
        )

    fairness_results["omnibus_tests"].to_csv(
        f"results/tables/{prefix}_fairness_omnibus_tests.csv",
        index=False,
    )

    fairness_results["pairwise_tests"].to_csv(
        f"results/tables/{prefix}_fairness_pairwise_tests.csv",
        index=False,
    )
