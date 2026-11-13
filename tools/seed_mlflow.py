"""Rebuild mlflow.db in the state expected at a given checkpoint.

    python tools/seed_mlflow.py --upto 1   # nothing: no tracking database yet
    python tools/seed_mlflow.py --upto 2   # idem
    python tools/seed_mlflow.py --upto 3   # the four runs of cycle 3
    python tools/seed_mlflow.py --upto 4   # idem + tags + registry (2 versions, @champion on 2)

Teacher tool, called by restore.sh. Students never run it themselves.

It retrains the models for real, with the same split and the same seed as
src/train.py, so the F1 values it stores are the ones a student sees.
"""

import argparse
import os
import shutil
import subprocess
from pathlib import Path

import pandas as pd
import yaml
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split

TRACKING_URI = "sqlite:///mlflow.db"
EXPERIMENT = "wine-quality"
REGISTERED_MODEL = "wine-quality"
RANDOM_STATE = 42
TEST_SIZE = 0.2

# Same four runs as the statement, one thing changing at a time.
RUNS = [
    {"n_estimators": 100, "max_depth": 4},
    {"n_estimators": 300, "max_depth": 4},
    {"n_estimators": 300, "max_depth": 8},
    {"n_estimators": 300, "max_depth": 16},
]

RAW = Path("data/raw/winequality-red.csv")
SOURCE = Path("data/source/winequality-red.csv")
POINTER = Path("data/raw/winequality-red.csv.dvc")
ARTIFACTS = Path("mlartifacts")
DB = Path("mlflow.db")


def wipe():
    """Remove every trace of a previous tracking database."""
    if DB.exists():
        DB.unlink()
    for directory in (ARTIFACTS, Path("mlruns")):
        if directory.exists():
            shutil.rmtree(directory)


def load_data():
    csv = RAW if RAW.exists() else SOURCE
    df = pd.read_csv(csv, sep=";")
    X = df.drop(columns=["quality"])
    y = (df["quality"] >= 6).astype(int)
    return X, y


def git_commit():
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL
        )
        return out.decode().strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def data_md5():
    if not POINTER.exists():
        return "unknown"
    out = yaml.safe_load(POINTER.read_text())["outs"][0]
    return out.get("md5") or out.get("hash", "unknown")


def seed(upto):
    import mlflow
    import mlflow.sklearn
    from mlflow import MlflowClient

    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT)

    tags = {}
    if upto >= 4:
        tags = {"git_commit": git_commit(), "data_md5": data_md5()}

    X, y = load_data()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
    )

    results = []
    for params in RUNS:
        with mlflow.start_run() as run:
            mlflow.log_params(params)
            if tags:
                mlflow.set_tags(tags)
            model = RandomForestClassifier(random_state=RANDOM_STATE, **params)
            model.fit(X_train, y_train)
            f1 = f1_score(y_test, model.predict(X_test))
            mlflow.log_metric("f1", f1)
            info = mlflow.sklearn.log_model(
                model, name="model",
                # MLflow 3.17 uses skops by default; tree models need this.
                skops_trusted_types=["sklearn.tree._tree.Tree"],
            )
            results.append(
                {"run_id": run.info.run_id, "model_id": info.model_id, "f1": f1, **params}
            )
            print(f"  run {run.info.run_id[:8]}  {params}  f1={f1:.4f}")

    if upto >= 4:
        client = MlflowClient()
        # Version 1: the first serious candidate. Version 2: the best run.
        first = results[1]
        best = max(results, key=lambda r: r["f1"])
        if best["run_id"] == first["run_id"]:
            best = max(
                (r for r in results if r["run_id"] != first["run_id"]),
                key=lambda r: r["f1"],
            )
        for rank, result in enumerate((first, best), start=1):
            # models:/<model_id> is the MLflow 3 address of a logged model.
            # runs:/<run_id>/model still works but warns -- a model is no longer
            # an artefact of its run.
            version = mlflow.register_model(
                f"models:/{result['model_id']}", REGISTERED_MODEL
            )
            print(f"  registered version {version.version} <- run {result['run_id'][:8]}"
                  f" (f1={result['f1']:.4f})")
            assert str(version.version) == str(rank), "unexpected version number"
        client.set_registered_model_alias(REGISTERED_MODEL, "champion", "2")
        print("  alias @champion -> version 2")

    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upto", type=int, required=True, choices=[1, 2, 3, 4])
    args = parser.parse_args()

    # The statement sets this by hand in cycle 3; make the tool independent of it.
    os.environ["MLFLOW_TRACKING_URI"] = TRACKING_URI

    wipe()
    if args.upto < 3:
        print(f"seed_mlflow: checkpoint {args.upto} -- no MLflow state expected")
        return

    print(f"seed_mlflow: rebuilding mlflow.db for checkpoint {args.upto}")
    seed(args.upto)


if __name__ == "__main__":
    main()
