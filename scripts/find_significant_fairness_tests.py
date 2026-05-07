from pathlib import Path
import argparse
import pandas as pd


def load_test_files(results_dir):
    results_dir = Path(results_dir)

    files = list(results_dir.glob("*fairness_omnibus_tests.csv"))
    files += list(results_dir.glob("*fairness_pairwise_tests.csv"))

    return files


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", default="results/tables")
    parser.add_argument("--alpha", type=float, default=0.05)
    args = parser.parse_args()

    rows = []

    for path in load_test_files(args.results_dir):
        df = pd.read_csv(path)

        if "p_value" not in df.columns:
            continue

        sig = df[df["p_value"] < args.alpha].copy()

        if sig.empty:
            continue

        sig.insert(0, "source_file", path.name)
        rows.append(sig)

    if not rows:
        print(f"No significant tests found at alpha = {args.alpha}")
        return

    significant = pd.concat(rows, ignore_index=True)

    output_path = Path(args.results_dir) / f"significant_fairness_tests_alpha_{args.alpha}.csv"
    significant.to_csv(output_path, index=False)

    print(f"\nSignificant fairness tests at alpha = {args.alpha}:")
    print(significant)

    print(f"\nSaved to {output_path}")


if __name__ == "__main__":
    main()