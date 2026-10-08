"""
evaluate.py

Aggregates the per-fold metrics.json files produced by train.py into
the mean +/- std dev summary format specified in Section 4.3, for
both conditions, plus a combined confusion-matrix comparison for the
failure analysis in your Discussion section.

Usage:
    python evaluate.py --outdir ../outputs --savedir ../outputs/summary
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def load_condition(outdir: Path, condition: str):
    path = outdir / condition / "all_folds_metrics.json"
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}. Run train.py for '{condition}' first.")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def summarise(fold_metrics, condition: str):
    macro_f1s = [m["macro_f1"] for m in fold_metrics]
    aucs = [m["macro_auc_roc"] for m in fold_metrics if not np.isnan(m["macro_auc_roc"])]
    accs = [m["accuracy"] for m in fold_metrics]

    rows = [{
        "condition": condition,
        "metric": "macro_f1",
        "mean": np.mean(macro_f1s),
        "std": np.std(macro_f1s, ddof=1),
        "per_fold": macro_f1s,
    }, {
        "condition": condition,
        "metric": "accuracy",
        "mean": np.mean(accs),
        "std": np.std(accs, ddof=1),
        "per_fold": accs,
    }, {
        "condition": condition,
        "metric": "macro_auc_roc",
        "mean": np.mean(aucs) if aucs else float("nan"),
        "std": np.std(aucs, ddof=1) if len(aucs) > 1 else float("nan"),
        "per_fold": aucs,
    }]

    # per-class summary
    for label in ["neutral", "offensive", "hate"]:
        for metric_name in ["precision", "recall", "f1"]:
            vals = [m["per_class"][label][metric_name] for m in fold_metrics]
            rows.append({
                "condition": condition,
                "metric": f"{label}_{metric_name}",
                "mean": np.mean(vals),
                "std": np.std(vals, ddof=1),
                "per_fold": vals,
            })
    return rows


def combined_confusion_matrix(fold_metrics):
    """Sum confusion matrices across all 5 folds for a single overall view."""
    cm_sum = np.zeros((3, 3), dtype=int)
    for m in fold_metrics:
        cm_sum += np.array(m["confusion_matrix"])
    return cm_sum


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", required=True, help="Same --outdir used in train.py")
    ap.add_argument("--savedir", required=True)
    args = ap.parse_args()

    outdir = Path(args.outdir)
    savedir = Path(args.savedir)
    savedir.mkdir(parents=True, exist_ok=True)

    all_rows = []
    matrices = {}
    for condition in ["no_emoji", "with_emoji"]:
        fold_metrics = load_condition(outdir, condition)
        all_rows.extend(summarise(fold_metrics, condition))
        matrices[condition] = combined_confusion_matrix(fold_metrics).tolist()

    df = pd.DataFrame(all_rows)
    df_display = df[["condition", "metric", "mean", "std"]].copy()
    df_display["mean"] = df_display["mean"].round(4)
    df_display["std"] = df_display["std"].round(4)

    csv_path = savedir / "summary_metrics.csv"
    df_display.to_csv(csv_path, index=False)

    json_path = savedir / "summary_metrics_full.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({"summary": df.to_dict(orient="records"), "confusion_matrices": matrices,
                    "confusion_matrix_label_order": ["neutral", "offensive", "hate"]}, f, indent=2)

    print("=== Macro-F1 (primary metric, Section 4.1) ===")
    print(df_display[df_display["metric"] == "macro_f1"].to_string(index=False))
    print("\n=== Full summary written to ===")
    print(csv_path)
    print(json_path)
    print("\nNext: run stats_test.py for the paired Wilcoxon significance test (Section 4.4).")


if __name__ == "__main__":
    main()
