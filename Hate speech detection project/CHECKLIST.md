# Checklist

## Done
- [x] Dataset loaded, cleaned, de-duplicated, conflicting labels removed
- [x] Preprocessing, aligned no-emoji / with-emoji variants, shared stratified folds
- [x] Training, evaluation and Wilcoxon scripts; Colab notebook

## To do
- [ ] Run `run_full_pipeline.ipynb` on Colab (cells 1-10); save the printed output and `outputs_final.zip`
- [ ] Check per-class F1 / confusion matrices; if a fold collapses to one class, retrain it for BOTH conditions with a different seed (state the rule in the paper)
- [ ] Label audit: `python 2_Model_Training/label_audit.py make ...`, two independent annotators, then `score`; report kappa
- [ ] Record annotation provenance (who, how many, guidelines, kappa) in the paper and DATA_CARD.md
- [ ] Fill paper/paper.tex TODOs with real numbers (Results, Wilcoxon, Limitations, Discussion)
- [ ] State Wilcoxon limit (min p = 0.0625 with 5 folds); do not claim "no effect" from a non-significant p
- [ ] Update the integrity declaration to match actual tool use; add an AI-use disclosure if required
- [ ] Push to GitHub (no keys, no .env), update repo link
- [ ] Journal submission files per paper/DISSEMINATION_PLAN.md
