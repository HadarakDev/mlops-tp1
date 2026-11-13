#!/usr/bin/env bash
#
# ./restore.sh N   brings this repository back to the state expected at the end
#                  of cycle N (N = 1..4), after putting your own work aside on a
#                  branch named mine-cN.
#
# Safe to run twice, and safe to run on a clean repository.
#
set -euo pipefail

N="${1:-}"
case "$N" in
  1 | 2 | 3 | 4) ;;
  *)
    echo "usage: ./restore.sh <1|2|3|4>" >&2
    exit 2
    ;;
esac

cd "$(dirname "$0")"
TAG="checkpoint-$N"
PYTHON="${PYTHON:-python}"
SOURCE_CSV="data/source/winequality-red.csv"
RAW_CSV="data/raw/winequality-red.csv"

if ! git rev-parse -q --verify "refs/tags/$TAG" >/dev/null; then
  echo "restore: tag $TAG not found -- is this the course repository?" >&2
  exit 1
fi

# --------------------------------------------------------------- 1. your work
BRANCH="$(git symbolic-ref --quiet --short HEAD || true)"
BRANCH="${BRANCH:-main}"

git add -A
if ! git diff --cached --quiet; then
  git commit -q -m "WIP before restore to $TAG"
  echo "restore: work in progress committed"
fi
if git rev-parse -q --verify HEAD >/dev/null; then
  git branch -f "mine-c$N" HEAD
  echo "restore: your work is kept on branch mine-c$N"
fi

# ---------------------------------------------------------------- 2. the code
git checkout -q -B "$BRANCH" "$TAG"
echo "restore: code back to $TAG (branch $BRANCH)"

# ---------------------------------------------------------------- 3. the data
mkdir -p data/raw
cp "$SOURCE_CSV" "$RAW_CSV"

if [ -d .dvc ]; then
  # Where the remote points, with a leading ~ expanded. Empty if none is set.
  STORE="$(dvc config remote.store.url 2>/dev/null || true)"
  case "$STORE" in
    "~" | "~/"*) STORE="$HOME${STORE#\~}" ;;
  esac

  # Cache both versions of the dataset, so that dvc checkout, dvc pull and
  # dvc push --all-commits have something to work with even after a
  # `rm -rf .dvc/cache`. Each version is cached -- and, from checkpoint 2 on,
  # pushed -- while it is the one the pointer describes.
  "$PYTHON" src/drop_rows.py --n 200 >/dev/null
  dvc add -q "$RAW_CSV"
  if [ "$N" -ge 2 ] && [ -n "$STORE" ]; then dvc push -q; fi

  cp "$SOURCE_CSV" "$RAW_CSV"
  dvc add -q "$RAW_CSV"
  if [ "$N" -ge 2 ] && [ -n "$STORE" ]; then dvc push -q; fi

  # Back to the pointer this checkpoint expects.
  git checkout -q -- "$RAW_CSV.dvc"
  dvc checkout -q --force "$RAW_CSV.dvc"
  echo "restore: cache refilled, $RAW_CSV back to $(($(wc -l <"$RAW_CSV") - 1)) rows"

  if [ "$N" -ge 2 ]; then
    if [ -n "$STORE" ]; then
      echo "restore: both dataset versions are on the remote ($STORE)"
    fi
  else
    # At the end of cycle 1 the remote is declared but nothing has been pushed
    # yet -- that is the whole point of cycle 2's counter. Empty it, so the
    # first `dvc push` really does report one object.
    if [ -n "$STORE" ] && [ -d "$STORE/files" ]; then
      rm -rf "$STORE/files"
      echo "restore: remote emptied ($STORE) -- nothing is pushed yet at checkpoint-1"
    fi
  fi
fi

# ---------------------------------------------------------------- 4. the runs
"$PYTHON" tools/seed_mlflow.py --upto "$N"

echo "restore: done -- you are at $TAG"
