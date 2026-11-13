"""Train a RandomForest on the red wine quality dataset.

Task: binary classification, "good wine" = quality >= 6. Metric: F1.
Every run is logged to MLflow, with the code commit and the data md5 as tags
(set MLFLOW_TRACKING_URI first).
"""

# Importing pandas and scikit-learn takes a few seconds: say so before
# it happens, otherwise the terminal looks frozen.
print("loading libraries...", flush=True)

import argparse
import sys
import subprocess
from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
import yaml
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split

DATA = Path("data/raw/winequality-red.csv")
MODEL = Path("models/model.pkl")
RANDOM_STATE = 42
TEST_SIZE = 0.2


def load_data():
    if not DATA.exists():
        sys.exit(
            f"{DATA} is missing.\n"
            "Put the dataset in place first:\n"
            "    cp data/source/winequality-red.csv data/raw/\n"
            "or, if DVC already tracks it:  dvc checkout"
        )
    df = pd.read_csv(DATA, sep=";")
    X = df.drop(columns=["quality"])
    y = (df["quality"] >= 6).astype(int)
    return X, y


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-estimators", type=int, default=100)
    parser.add_argument("--max-depth", type=int, default=None)
    return parser.parse_args()


def main():
    args = parse_args()

    print(f"reading {DATA}", flush=True)
    X, y = load_data()
    print(f"{len(X)} rows, training {args.n_estimators} trees "
          f"(max_depth={args.max_depth})...", flush=True)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
    )

    model = RandomForestClassifier(
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        random_state=RANDOM_STATE,
    )

    mlflow.set_experiment("wine-quality")

    commit = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"]).decode().strip()
    data_md5 = yaml.safe_load(open("data/raw/winequality-red.csv.dvc"))["outs"][0]["md5"]

    with mlflow.start_run():
        mlflow.log_params({"n_estimators": args.n_estimators,
                           "max_depth": args.max_depth})
        mlflow.set_tags({"git_commit": commit, "data_md5": data_md5})
        model.fit(X_train, y_train)
        f1 = f1_score(y_test, model.predict(X_test))
        mlflow.log_metric("f1", f1)
        # MLflow 3.17 serialises sklearn models with skops by default, and skops
        # refuses tree-based models unless their node type is declared trusted.
        info = mlflow.sklearn.log_model(model, name="model",
                                        skops_trusted_types=["sklearn.tree._tree.Tree"])

    print(f"f1 = {f1:.4f}")
    # The address to register this model under: mlflow.register_model(f"models:/{model_id}", ...)
    print(f"model_id = {info.model_id}")

    MODEL.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL)
    print(f"saved {MODEL}")


if __name__ == "__main__":
    main()
