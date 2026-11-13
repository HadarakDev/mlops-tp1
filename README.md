# MLOps -- TP 1: version your data, trace your runs

Red wine quality, as a binary task: a wine is *good* when `quality >= 6`.
Metric: F1. Model: RandomForest, `random_state=42`.

## Setup

Python **3.12 or newer** is required (numpy 2.5 and scikit-learn 1.9 ask for it).
Check yours with `python3 -V` before anything else.

With venv:

```bash
python3.12 -m venv .venv && source .venv/bin/activate
```

With conda:

```bash
conda create -y -n mlops-tp1 python=3.12
conda activate mlops-tp1
```

Then, either way:

```bash
pip install -r requirements.txt
python -V                                      # 3.12 or newer
dvc --version && mlflow --version
cp data/source/winequality-red.csv data/raw/
```

If pip says `No matching distribution found for mlflow==3.17.1`, your `pip` runs
on an older Python — most likely a conda `base` environment that was never
deactivated.

## Scripts

| Command | What it does |
|---|---|
| `python src/train.py --n-estimators 300 --max-depth 8` | trains, prints the F1, writes `models/model.pkl` |
| `python src/predict.py --sample data/samples/one_wine.json` | predicts one wine |
| `python src/drop_rows.py --n 200` | drops 200 rows from `data/raw/`, deterministically |

## Lost? Back to the state of the class

```bash
./restore.sh 2
```

Your own work is kept on a `mine-c2` branch; the code, the data and the MLflow
runs go back to the reference state of the end of cycle 2. Using it is neither
cheating nor an admission of defeat.

## What you hand in

`REPORT.md` filled in (Q0 to Q4), the repository with an up-to-date `.dvc`
pointer, and two screenshots: your four runs compared, and the registry with its
two versions.
