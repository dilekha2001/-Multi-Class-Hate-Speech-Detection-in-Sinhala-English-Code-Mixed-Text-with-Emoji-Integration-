"""
make_folds.py

Creates the Stratified 5-Fold split described in Section 4.3
(Validation Strategy) of the Milestone 2 report. The SAME fold
assignment is reused for both the baseline (no-emoji) and proposed
(with-emoji) models, which is required for the paired Wilcoxon
signed-rank test in Section 4.4 to be statistically valid.

IMPORTANT — duplicate-text leakage guard:
Real-world scraped data (and some synthetic test sets) can contain the
exact same comment text repeated many times. A plain StratifiedKFold
would happily place identical text in both the train and test portions
of a fold, which leaks test information into training and inflates
F1 in a way that does not reflect real generalisation. This script
therefore uses **StratifiedGroupKFold**, grouping on the normalised
comment text, so that all copies of a given comment always land in the
same fold. Run the --check_duplicates flag first to see how bad this
is for your data.

This script identifies rows by their dataframe row position (a stable
0..N-1 index), NOT by `comment_id`, because `comment_id` is not
guaranteed to be unique in every source file (check this yourself --
duplicate IDs will silently break any join that assumes uniqueness).

Usage:
    # 1. Check duplication severity first
    python make_folds.py --labels_csv ../data/data_with_emoji.csv \
        --outdir ../data --check_duplicates_only

    # 2. Generate folds (group-split on comment text to prevent leakage)
    python make_folds.py --labels_csv ../data/data_with_emoji.csv \
        --outdir ../data --n_splits 5 --seed 42
"""

import argparse
import json
from pathlib import Path

import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold


def report_duplicates(df: pd.DataFrame) -> None:
    n_rows = len(df)
    n_unique_text = df["comment"].nunique()
    n_unique_id = df["comment_id"].nunique() if "comment_id" in df.columns else None
    print(f"Total rows: {n_rows}")
    print(f"Unique comment texts: {n_unique_text} ({100 * n_unique_text / n_rows:.1f}% unique)")
    if n_unique_id is not None:
        print(f"Unique comment_id values: {n_unique_id}")
        if n_unique_id < n_rows:
            print("  WARNING: comment_id is NOT unique per row -- do not use it as a row key.")
    if n_unique_text < n_rows:
        top = df["comment"].value_counts().head(5)
        print("  WARNING: duplicate comment text detected. Most repeated:")
        print(top.to_string())
        print("  -> Using StratifiedGroupKFold on comment text to prevent train/test leakage.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels_csv", required=True,
                     help="Any one of the processed CSVs (with/no emoji) -- only "
                          "the comment and label columns are used for fold assignment.")
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--n_splits", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--check_duplicates_only", action="store_true",
                     help="Print duplication diagnostics and exit without writing folds.")
    args = ap.parse_args()

    df = pd.read_csv(args.labels_csv, encoding="utf-8-sig").reset_index(drop=True)
    report_duplicates(df)
    if args.check_duplicates_only:
        return

    row_ids = df.index.tolist()          # stable 0..N-1 row identifiers
    labels = df["label"].tolist()
    groups = df["comment"].str.strip().str.lower().tolist()  # group by normalised text

    sgkf = StratifiedGroupKFold(n_splits=args.n_splits, shuffle=True, random_state=args.seed)

    folds = []
    for fold_idx, (train_idx, test_idx) in enumerate(sgkf.split(row_ids, labels, groups)):
        train_rows = [int(i) for i in train_idx]
        test_rows = [int(i) for i in test_idx]
        folds.append({
            "fold": fold_idx + 1,
            "train_rows": train_rows,
            "test_rows": test_rows,
            "n_train": len(train_rows),
            "n_test": len(test_rows),
        })
        test_labels = df.iloc[test_rows]["label"].value_counts().to_dict()
        leak_check = set(df.iloc[train_rows]["comment"]) & set(df.iloc[test_rows]["comment"])
        print(f"Fold {fold_idx + 1}: train={len(train_rows)}, test={len(test_rows)}, "
              f"test label distribution={test_labels}, "
              f"leaked duplicate texts across split={len(leak_check)}")

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    out_path = outdir / "folds.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({
            "n_splits": args.n_splits,
            "seed": args.seed,
            "row_key": "dataframe row position (0-indexed) in the CSV used to build folds",
            "folds": folds,
        }, f, indent=2)

    print(f"\nWrote fold assignments to {out_path}")
    print("NOTE: train.py must load the SAME csv (same row order) to align these row positions.")


if __name__ == "__main__":
    main()
