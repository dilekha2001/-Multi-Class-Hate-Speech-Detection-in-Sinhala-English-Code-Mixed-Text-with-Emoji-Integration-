"""
annotation_agreement.py

Takes a CSV where 3 annotators have each independently labelled every
comment, computes pairwise Cohen's Kappa (Section 1.4), applies
majority-vote labelling, and flags rows with no consensus (all 3
different) for either discarding or re-annotation per your protocol:
"Annotations that do not have consensus among the three annotators
will be ignored."

Expected input CSV columns:
    comment_id, comment, source, annotator_1, annotator_2, annotator_3
where each annotator_N column contains one of: hate, offensive, neutral

Usage:
    python annotation_agreement.py \
        --input ../../data/annotation_sheet.csv \
        --outfile ../../data/annotated_final.csv \
        --min_kappa 0.70
"""

import argparse
from collections import Counter

import pandas as pd
from sklearn.metrics import cohen_kappa_score


def majority_vote(row, annotator_cols):
    votes = [row[c] for c in annotator_cols]
    counts = Counter(votes)
    top_label, top_count = counts.most_common(1)[0]
    if top_count >= 2:  # at least 2 of 3 agree
        return top_label
    return None  # all three disagree -> no consensus


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--outfile", required=True)
    ap.add_argument("--annotator_cols", default="annotator_1,annotator_2,annotator_3")
    ap.add_argument("--min_kappa", type=float, default=0.70,
                     help="Threshold from Section 1.4: if any pairwise kappa "
                          "falls below this, guidelines should be revised and "
                          "affected batches re-annotated.")
    args = ap.parse_args()

    annotator_cols = args.annotator_cols.split(",")
    df = pd.read_csv(args.input, encoding="utf-8-sig")

    missing = [c for c in annotator_cols if c not in df.columns]
    if missing:
        raise SystemExit(f"Missing expected annotator columns: {missing}")

    valid_labels = {"hate", "offensive", "neutral"}
    for c in annotator_cols:
        bad = set(df[c].dropna().unique()) - valid_labels
        if bad:
            raise SystemExit(f"Column {c} has unexpected label values: {bad}. "
                              f"Expected only: {valid_labels}")

    print("=== Pairwise Cohen's Kappa ===")
    pairs = [(0, 1), (0, 2), (1, 2)]
    kappas = []
    for i, j in pairs:
        k = cohen_kappa_score(df[annotator_cols[i]], df[annotator_cols[j]])
        kappas.append(k)
        print(f"  {annotator_cols[i]} vs {annotator_cols[j]}: kappa = {k:.3f}")

    mean_kappa = sum(kappas) / len(kappas)
    print(f"\nMean pairwise kappa: {mean_kappa:.3f}")
    if mean_kappa < args.min_kappa:
        print(f"BELOW THRESHOLD ({args.min_kappa}) -- per Section 1.4, revise annotation "
              f"guidelines and re-annotate the lowest-agreement batches before proceeding.")
    else:
        print(f"Meets the {args.min_kappa} threshold from Section 1.4.")

    # majority vote + consensus check
    df["label"] = df.apply(lambda r: majority_vote(r, annotator_cols), axis=1)
    no_consensus = df["label"].isna().sum()
    print(f"\nRows with no consensus (all 3 annotators disagree): {no_consensus} "
          f"({100*no_consensus/len(df):.1f}%) -- these are DROPPED per your protocol.")

    df_final = df[df["label"].notna()].drop(columns=annotator_cols)
    print(f"\nFinal labelled rows: {len(df_final)}")
    print(df_final["label"].value_counts())

    df_final.to_csv(args.outfile, index=False, encoding="utf-8-sig")
    print(f"\nWrote {args.outfile}")
    print("This file is now ready for preprocessing.py.")


if __name__ == "__main__":
    main()
