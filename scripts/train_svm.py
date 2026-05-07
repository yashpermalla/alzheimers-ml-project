
from scripts._common import run_model_script
from src.train import fit_svm


if __name__ == "__main__":
    run_model_script("svm", fit_svm)
