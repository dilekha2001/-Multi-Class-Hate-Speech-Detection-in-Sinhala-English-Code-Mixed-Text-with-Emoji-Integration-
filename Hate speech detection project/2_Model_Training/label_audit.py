"""
label_audit.py -- blind re-annotation audit of a random sample of labelled comments.

  python label_audit.py make  --data ../data/annotated_final.csv --outdir ../data/label_audit
      -> audit_sheet_for_annotators.csv  (blind: no labels, no platform)
         audit_KEY_do_not_share.csv      (existing labels; keep private)
  (give each annotator their own copy with only their columns, collect them back,
   merge into audit_sheet_for_annotators.csv with columns annotator_A / annotator_B)
  python label_audit.py score --outdir ../data/label_audit
      -> Cohen's kappa: A vs B, A/B vs existing labels, confusion table, by platform.
"""
import argparse, pandas as pd
from pathlib import Path
from sklearn.metrics import cohen_kappa_score, confusion_matrix

def make(a):
    out = Path(a.outdir); out.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(a.data, encoding="utf-8-sig")
    parts = [g.sample(n=min(a.n_per_cell, len(g)), random_state=a.seed) for _, g in df.groupby(["source", "label"])]
    s = pd.concat(parts).sample(frac=1, random_state=a.seed).reset_index(drop=True)
    s["audit_id"] = [f"A{i+1:03d}" for i in range(len(s))]
    s[["audit_id", "comment_id", "source", "label"]].to_csv(out / "audit_KEY_do_not_share.csv", index=False, encoding="utf-8-sig")
    b = s[["audit_id", "comment"]].copy()
    for c in ["annotator_A", "annotator_B", "targets_group_A", "targets_group_B"]:
        b[c] = ""
    b.to_csv(out / "audit_sheet_for_annotators.csv", index=False, encoding="utf-8-sig")
    print(len(b), "comments written to", out)

def score(a):
    out = Path(a.outdir)
    d = pd.read_csv(out / "audit_sheet_for_annotators.csv", encoding="utf-8-sig").merge(
        pd.read_csv(out / "audit_KEY_do_not_share.csv", encoding="utf-8-sig"), on="audit_id")
    for c in ["annotator_A", "annotator_B"]:
        d[c] = d[c].astype(str).str.strip().str.lower()
    ok = {"hate", "offensive", "neutral"}
    d = d[d.annotator_A.isin(ok) & d.annotator_B.isin(ok)]
    k = lambda x, y: round(cohen_kappa_score(x, y), 3)
    print("kappa A vs B:", k(d.annotator_A, d.annotator_B))
    print("kappa A vs existing:", k(d.annotator_A, d.label), "| B vs existing:", k(d.annotator_B, d.label))
    ag = d[d.annotator_A == d.annotator_B]
    order = ["neutral", "offensive", "hate"]
    print("A=B on", len(ag), "of", len(d), "| kappa consensus vs existing:", k(ag.annotator_A, ag.label))
    print(pd.DataFrame(confusion_matrix(ag.label, ag.annotator_A, labels=order), index=order, columns=order))
    for s, g in d.groupby("source"):
        print(s, "kappa A vs existing:", k(g.annotator_A, g.label))

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); sp = ap.add_subparsers(dest="cmd", required=True)
    m = sp.add_parser("make"); m.add_argument("--data", required=True); m.add_argument("--outdir", required=True)
    m.add_argument("--n_per_cell", type=int, default=10); m.add_argument("--seed", type=int, default=2026)
    s = sp.add_parser("score"); s.add_argument("--outdir", required=True)
    a = ap.parse_args(); (make if a.cmd == "make" else score)(a)
