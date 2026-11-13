"""Predict whether one wine is good, from a JSON sample."""

# Importing pandas and scikit-learn takes a few seconds: say so before
# it happens, otherwise the terminal looks frozen.
print("loading libraries...", flush=True)

import argparse
import json
from pathlib import Path

import joblib
import pandas as pd

MODEL = Path("models/model.pkl")

# The model was trained on these columns, in this order.
FEATURES = [
    "fixed acidity",
    "volatile acidity",
    "citric acid",
    "residual sugar",
    "chlorides",
    "free sulfur dioxide",
    "total sulfur dioxide",
    "density",
    "pH",
    "sulphates",
    "alcohol",
]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample", default="data/samples/one_wine.json")
    return parser.parse_args()


def load_model():
    return joblib.load(MODEL)


def main():
    args = parse_args()
    sample = json.loads(Path(args.sample).read_text())
    X = pd.DataFrame([sample])[FEATURES]

    model = load_model()
    prediction = int(model.predict(X)[0])
    proba = float(model.predict_proba(X)[0][1])

    print(f"sample     : {args.sample}")
    print(f"good wine  : {bool(prediction)}")
    print(f"p(good)    : {proba:.3f}")


if __name__ == "__main__":
    main()
