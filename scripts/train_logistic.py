
from scripts._common import run_model_script
from src.train import fit_logistic_regression


if __name__ == "__main__":
    run_model_script("logistic_regression", fit_logistic_regression)
