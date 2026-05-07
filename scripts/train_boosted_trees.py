
from scripts._common import run_model_script
from src.train import fit_boosted_trees


if __name__ == "__main__":
    run_model_script("boosted_trees", fit_boosted_trees)
