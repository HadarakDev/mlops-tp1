#!/usr/bin/env bash
#
# Teacher tool. Rebuilds the whole Git history of the reference repository and
# the four checkpoint-N tags from the files in the working tree.
#
# Run it after changing src/, tools/ or the statement:
#
#     PATH=/path/to/venv/bin:$PATH ./tools/build_checkpoints.sh
#
# It rewrites .git, so anything not in the working tree is lost.
#
set -euo pipefail
cd "$(dirname "$0")/.."

for tool in git dvc python; do
  command -v "$tool" >/dev/null || { echo "missing: $tool" >&2; exit 1; }
done

rm -rf .git .dvc .dvcignore models mlflow.db mlartifacts
rm -f data/raw/winequality-red.csv data/raw/winequality-red.csv.dvc data/raw/.gitignore

git init -q -b main
git config user.name "Nicolas Roche"
git config user.email "nicolasroche.contact@gmail.com"
git add -A
git commit -q -m "Skeleton: training script, prediction script, dataset and restore script"

git checkout -q -b reference

# ---- cycle 1: DVC, the remote, the dataset, a second version, back to the first.
#      The remote is declared here, before any data commit: DVC resolves it per
#      revision, so a remote added later would make `dvc push --all-commits` skip
#      the earlier commits in silence. '~/dvc-store' is stored unexpanded on
#      purpose, so the committed config works on every student's machine.
cp data/source/winequality-red.csv data/raw/
dvc init -q
dvc remote add -d -q store '~/dvc-store'
dvc add -q data/raw/winequality-red.csv
git add data/raw/winequality-red.csv.dvc data/raw/.gitignore .dvc/config .dvcignore
git commit -q -m "Track raw dataset with DVC and declare the remote"
python src/drop_rows.py --n 200 >/dev/null
dvc add -q data/raw/winequality-red.csv
git commit -q -am "Drop 200 rows from the raw dataset"
git checkout -q HEAD~1 -- data/raw/winequality-red.csv.dvc
dvc checkout -q --force
git commit -q -am "Restore the full dataset version"
git tag -f checkpoint-1

# ---- cycle 2 changes no file: it is push, pull and cache work. So checkpoint-2
#      is the same commit as checkpoint-1; what tells them apart is the remote,
#      which restore.sh fills (checkpoint-2) or empties (checkpoint-1).
git tag -f checkpoint-2

# ---- cycle 3: train.py logs to MLflow
cp tools/checkpoints/train_c3.py src/train.py
git commit -q -am "Log every training run to MLflow"
git tag -f checkpoint-3

# ---- cycle 4: tags on the run, and a model served by alias
cp tools/checkpoints/train_c4.py src/train.py
cp tools/checkpoints/predict_c4.py src/predict.py
git commit -q -am "Tag runs with code commit and data md5, serve the model by alias"
git tag -f checkpoint-4

# ---- back to the state a student clones
git checkout -q main
git branch -D reference >/dev/null
rm -rf .dvc .dvcignore data/raw/winequality-red.csv

git log --oneline --all --decorate
git status --short --ignored
echo "build: done"
