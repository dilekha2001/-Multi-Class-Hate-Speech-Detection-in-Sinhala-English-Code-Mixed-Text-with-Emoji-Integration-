"""
stats_test.py

Runs the paired Wilcoxon signed-rank test on the 5 per-fold macro-F1
scores of the baseline (no-emoji) vs proposed (with-emoji) model, as
specified in Section 4.4 (Statistical Significance Test). This is
your falsification test for the Research Question stated in
Assignment 1.

Requires that both conditions were trained on the EXACT SAME folds
(guaranteed if you used the same folds.json for both train.py runs).

Usage:
    python stats_test.py --outdir ../outputs --savedir ../outputs/summary
"""

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon, shapiro


def load_fold_f1s(outdir: Path, condition: str):
    path = outdir / condition / "all_folds_metrics.json"
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    data_sorted = sorted(data, key=lambda m: m["fold"])
    return [m["macro_f1"] for m in data_sorted], [m["fold"] for m in data_sorted]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--savedir", required=True)
    ap.add_argument("--alpha", type=float, default=0.05)
    args = ap.parse_args()

    outdir = Path(args.outdir)
    savedir = Path(args.savedir)
    savedir.mkdir(parents=True, exist_ok=True)

    baseline_f1, baseline_folds = load_fold_f1s(outdir, "no_emoji")
    proposed_f1, proposed_folds = load_fold_f1s(outdir, "with_emoji")

    if baseline_folds != proposed_folds:
        raise ValueError("Fold order mismatch between conditions -- re-run with matching folds.json.")

    baseline_f1 = np.array(baseline_f1)
    proposed_f1 = np.array(proposed_f1)
    diffs = proposed_f1 - baseline_f1

    print("Fold | baseline (no emoji) | proposed (with emoji) | diff")
    for k, b, p, d in zip(baseline_folds, baseline_f1, proposed_f1, diffs):
        print(f"  {k}  |        {b:.4f}        |        {p:.4f}         | {d:+.4f}")

    print(f"\nMean baseline macro-F1: {baseline_f1.mean():.4f} +/- {baseline_f1.std(ddof=1):.4f}")
    print(f"Mean proposed macro-F1: {proposed_f1.mean():.4f} +/- {proposed_f1.std(ddof=1):.4f}")
    print(f"Mean difference (proposed - baseline): {diffs.mean():+.4f}")

    # Wilcoxon signed-rank test (paired, non-parametric) -- Section 4.4
    if np.all(diffs == 0):
        stat, p_value = float("nan"), 1.0
        print("\nAll per-fold differences are exactly zero -- Wilcoxon test is undefined.")
    else:
        stat, p_value = wilcoxon(proposed_f1, baseline_f1)

    print(f"\nWilcoxon signed-rank test: W = {stat}, p = {p_value:.4f}")
    significant = p_value < args.alpha
    if significant:
        direction = "IMPROVES" if diffs.mean() > 0 else "WORSENS"
        print(f"Result: statistically significant at alpha={args.alpha} "
              f"-> emoji encoding {direction} macro-F1.")
    else:
        print(f"Result: NOT statistically significant at alpha={args.alpha} "
              f"-> cannot reject the null hypothesis that emoji encoding has no effect. "
              f"This is a valid, reportable outcome per your falsifiable research question.")

    # With only n=5 paired observations, note the low statistical power explicitly
    # (5! = 120 possible sign/rank permutations caps the smallest achievable p-value)
    print("\nNote: with n=5 paired folds, the Wilcoxon test has limited power -- "
          "the smallest possible two-sided p-value is 0.0625. Report this caveat "
          "alongside your result rather than over-interpreting a non-significant p-value "
          "as proof of 'no effect'.")

    out = {
        "baseline_macro_f1_per_fold": baseline_f1.tolist(),
        "proposed_macro_f1_per_fold": proposed_f1.tolist(),
        "baseline_mean": float(baseline_f1.mean()),
        "baseline_std": float(baseline_f1.std(ddof=1)),
        "proposed_mean": float(proposed_f1.mean()),
        "proposed_std": float(proposed_f1.std(ddof=1)),
        "mean_diff_proposed_minus_baseline": float(diffs.mean()),
        "wilcoxon_statistic": float(stat) if stat == stat else None,  # NaN check
        "wilcoxon_p_value": float(p_value),
        "alpha": args.alpha,
        "significant": bool(significant),
    }
    out_path = savedir / "wilcoxon_result.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved result to {out_path}")


if __name__ == "__main__":
    main()
