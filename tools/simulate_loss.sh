#!/usr/bin/env bash
#
# Simulate losing your local data: removes the working file and the DVC cache.
# Your pointers stay in Git, so everything comes back with a dvc pull.
#
set -euo pipefail
cd "$(dirname "$0")/.."

rm -rf data/raw/winequality-red.csv .dvc/cache

echo "simulate_loss: data/raw/winequality-red.csv is gone"
echo "simulate_loss: .dvc/cache is gone"
echo "simulate_loss: the pointers are still in Git -- now try to get the data back"
