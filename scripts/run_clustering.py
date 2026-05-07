import argparse

import pandas as pd
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from sklearn.mixture import GaussianMixture

from src.data import get_train_test_data, parse_features
from src.train import make_preprocessor
from src.config import RANDOM_STATE


def to_dense(X):
    if hasattr(X, "toarray"):
        return X.toarray()
    return X


def evaluate_clustering(y_true, labels):
    return {
        "ARI": adjusted_rand_score(y_true, labels),
        "NMI": normalized_mutual_info_score(y_true, labels),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--feature-set", choices=["selected", "all", "custom"], default="selected")
    parser.add_argument("--features", type=str, default=None)
    parser.add_argument("--data-path", type=str, default=None)
    parser.add_argument("--n-clusters", type=int, default=2)
    args = parser.parse_args()

    custom_features = parse_features(args.features)

    X_train, X_test, y_train, y_test, feature_names = get_train_test_data(
        feature_set=args.feature_set,
        custom_features=custom_features,
        data_path=args.data_path,
    )

    X = pd.concat([X_train, X_test], axis=0)
    y = pd.concat([y_train, y_test], axis=0)

    preprocessor = make_preprocessor(feature_names)
    X_processed = to_dense(preprocessor.fit_transform(X))

    results = []

    kmeans = KMeans(
        n_clusters=args.n_clusters,
        random_state=RANDOM_STATE,
        n_init=20,
    )
    kmeans_labels = kmeans.fit_predict(X_processed)
    results.append({
        "method": "KMeans",
        **evaluate_clustering(y, kmeans_labels),
    })

    agglomerative = AgglomerativeClustering(
        n_clusters=args.n_clusters,
    )
    agglomerative_labels = agglomerative.fit_predict(X_processed)
    results.append({
        "method": "Agglomerative",
        **evaluate_clustering(y, agglomerative_labels),
    })

    gmm = GaussianMixture(
        n_components=args.n_clusters,
        random_state=RANDOM_STATE,
    )
    gmm_labels = gmm.fit_predict(X_processed)
    results.append({
        "method": "GaussianMixture",
        **evaluate_clustering(y, gmm_labels),
    })

    results_df = pd.DataFrame(results)

    print("\nClustering results")
    print(results_df)

    output_path = f"results/tables/clustering_{args.feature_set}.csv"
    results_df.to_csv(output_path, index=False)

    print(f"\nSaved results to {output_path}")


if __name__ == "__main__":
    main()