"""Train a RandomForest on the red wine quality dataset.

Task: binary classification, "good wine" = quality >= 6. Metric: F1.
"""

# Importing pandas and scikit-learn takes a few seconds: say so before
# it happens, otherwise the terminal looks frozen.
print("loading libraries...", flush=True)

import argparse
import sys
from pathlib import Path

import joblib
import pandas as pd
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
    model.fit(X_train, y_train)
    f1 = f1_score(y_test, model.predict(X_test))

    print(f"f1 = {f1:.4f}")

    MODEL.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL)
    print(f"saved {MODEL}")


if __name__ == "__main__":
    main()
