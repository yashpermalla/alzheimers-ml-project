from pathlib import Path

import kagglehub
import pandas as pd
from sklearn.model_selection import train_test_split

from src.config import (
    BINARY_FEATURES,
    CATEGORICAL_FEATURES,
    CONTINUOUS_FEATURES,
    ID_COLUMNS,
    RANDOM_STATE,
    SELECTED_FEATURES,
    TARGET,
)


def load_data(data_path=None):
    if data_path is None:
        dataset_dir = Path(
            kagglehub.dataset_download(
                "rabieelkharoua/alzheimers-disease-dataset"
            )
        )
        data_path = dataset_dir / "alzheimers_disease_data.csv"

    return pd.read_csv(data_path)


def parse_features(features_string):
    if features_string is None:
        return None

    return [
        feature.strip()
        for feature in features_string.split(",")
        if feature.strip()
    ]


def choose_features(df, feature_set="selected", custom_features=None):
    if feature_set == "selected":
        return [
            feature for feature in SELECTED_FEATURES
            if feature in df.columns
        ]

    if feature_set == "all":
        return [
            col for col in df.columns
            if col not in ID_COLUMNS + [TARGET]
        ]

    if feature_set == "custom":
        if not custom_features:
            raise ValueError(
                "For custom features, pass --features feature1,feature2,..."
            )

        missing = [
            feature for feature in custom_features
            if feature not in df.columns
        ]

        if missing:
            raise ValueError(
                f"These custom features are missing from the data: {missing}"
            )

        return custom_features

    raise ValueError("feature_set must be selected, all, or custom")


def feature_groups(feature_names):
    continuous = []
    categorical = []
    binary = []

    for feature in feature_names:
        if feature in CATEGORICAL_FEATURES:
            categorical.append(feature)
        elif feature in BINARY_FEATURES:
            binary.append(feature)
        elif feature in CONTINUOUS_FEATURES:
            continuous.append(feature)
        else:
            continuous.append(feature)

    return continuous, categorical, binary


def get_train_test_data(
    feature_set="selected",
    custom_features=None,
    exclude_features=None,
    data_path=None,
):
    df = load_data(data_path)

    all_selected_features = choose_features(
        df,
        feature_set=feature_set,
        custom_features=custom_features,
    )

    if exclude_features:
        missing_exclusions = [
            feature for feature in exclude_features
            if feature not in all_selected_features
        ]

        if missing_exclusions:
            raise ValueError(
                "These excluded features are not in the chosen feature set: "
                f"{missing_exclusions}"
            )

        model_features = [
            feature for feature in all_selected_features
            if feature not in exclude_features
        ]
    else:
        model_features = all_selected_features

    X = df[all_selected_features].copy()
    y = df[TARGET].copy()

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        stratify=y,
        random_state=RANDOM_STATE,
    )

    return X_train, X_test, y_train, y_test, model_features