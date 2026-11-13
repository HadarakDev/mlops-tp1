"""Drop N rows from the raw dataset, in place and deterministically.

Rows are picked with a fixed seed and the file is rewritten line by line, so
every student ends up with the exact same bytes -- and the same md5 in the
.dvc pointer.
"""

import argparse
from pathlib import Path

import numpy as np

CSV = Path("data/raw/winequality-red.csv")
SEED = 42


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=200, help="number of rows to drop")
    return parser.parse_args()


def main():
    args = parse_args()

    lines = CSV.read_text(encoding="utf-8").splitlines(keepends=True)
    header, rows = lines[0], lines[1:]
    if args.n > len(rows):
        raise SystemExit(f"cannot drop {args.n} rows out of {len(rows)}")

    rng = np.random.default_rng(SEED)
    dropped = set(rng.choice(len(rows), size=args.n, replace=False).tolist())
    kept = [row for index, row in enumerate(rows) if index not in dropped]

    CSV.write_text(header + "".join(kept), encoding="utf-8")
    print(f"{CSV}: {len(rows)} -> {len(kept)} rows ({args.n} dropped, seed={SEED})")


if __name__ == "__main__":
    main()
