# Multi-Class Hate Speech Detection in Sinhala-English Code-Mixed Text with Emoji Integration (XLM-RoBERTa)



IT41043 Intelligent Systems - Horizon Campus. Research question: does encoding emoji as text improve
macro-F1 of fine-tuned XLM-RoBERTa on three-class (neutral / offensive / hate) Facebook + YouTube comments?

Result (verified Colab T4 run, 5 paired folds): proposed **0.6786 +/- 0.0291** vs baseline **0.5836 +/- 0.0122**
(mean gain **+0.0950**, wins **5/5 folds**, Wilcoxon **W=0.0 p=0.0625** - minimum possible with n=5, so report
consistency 5/5, not just p).

## 60-second demo (laptop, no GPU)

```bash
pip install --break-system-packages streamlit pandas scikit-learn emoji matplotlib seaborn
python3 -m streamlit run 4_Presentation_and_App/app.py
# open http://localhost:8501
```

- Tab 1 Live Check: type a Sinhala-English comment with emoji (e.g. `watti amma kenek 😂`),
  toggle `with_emoji` (proposed: 😡 -> `angry face`) vs `no_emoji` (baseline: emoji removed),
  press Predict. Live model is TF-IDF char_wb(2,5) + LogisticRegression trained on 1947 rows
  (same as notebook Cell 9) - fast on CPU. If the prediction/confidence shifts between toggles,
  that is the research effect live.
- Tab 2 Real Proof: static XLM-RoBERTa 5-fold results loaded from local `outputs_final/summary/`
  (bar chart per-fold wins, per-class table, 2x summed confusion heatmaps, Wilcoxon note).
- Tab 3 Data + Method: dataset breakdown and pipeline for marking.

Note on Ubuntu `externally-managed-environment` (PEP 668): use `--break-system-packages` as above,
or `pip install --user`. A `python3 -m venv` needs `sudo apt install python3.14-venv` (terminal auth).

## Folder layout

```
1_Data_Preparation/
  data_raw/YOUTUBE_FACEBOOK_DATA.xlsx   original dataset (kept for provenance, do not edit, 2419 rows)
  annotated_final.csv                   cleaned dataset (2,210 rows): comment_id, comment, label, source
  DATA_CARD.md                          dataset description (use for the paper's Data section)
  cleaning_report.json                  what was removed and why (-3 null, -15 empty, -87 conflict, -105 dup)
  preprocessing.py                      7-step pipeline + emoji encode/strip toggle + common-row guard
  preprocessing_report.json             drop counts, 99.95% <= 128 tokens (justifies max_len=128)
  data_processed/
    data_no_emoji.csv                   baseline input  (emoji stripped, 1947 rows, same order)
    data_with_emoji.csv                 proposed input  (emoji -> text description, 1947 rows, same order)
    folds.json                          stratified 5-fold split (seed 42), shared by both conditions
2_Model_Training/
  train.py  dataset.py  make_folds.py  label_audit.py   XLM-R fine-tuning + folds + audit
  collection/                           youtube_collect.py, merge_and_scrub.py, annotation_agreement.py
3_Evaluation_and_Stats/
  evaluate.py  stats_test.py            mean±std + confusion matrices + Wilcoxon (min p=0.0625, n=5)
  outputs_final/                        local proof (gitignored, keep zip backup separately)
    no_emoji|with_emoji/fold_*/metrics.json + all_folds_metrics.json
    summary/summary_metrics.csv, summary_metrics_full.json, wilcoxon_result.json
  summary_metrics.csv                   copy of the headline table for quick reference
4_Presentation_and_App/
  app.py                                Streamlit 3-tab dashboard (portable: only Hate speech detection project/ paths)
  paper.tex + cover + dissemination plan (IEEE Access submission)
  2 report PDFs                         proposal + methodology (background)
run_full_pipeline.ipynb          Colab notebook Cells 1-10: train both models, evaluate, Wilcoxon, TF-IDF, zip
CHECKLIST.md                     what is done vs left to do
README.md                        this file
.gitignore                       ignores .env, __pycache__/, outputs*/, *.zip, *.bin/safetensors
```

## Results (from `outputs_final/summary/`)

| Condition | Macro-F1 | Accuracy | Macro AUC |
|---|---|---|---|
| Baseline no_emoji | 0.5836 +/- 0.0122 | 0.5948 | 0.7673 |
| Proposed with_emoji | 0.6786 +/- 0.0291 | 0.6908 | 0.8526 |

Per-class F1: offensive 0.4155 -> 0.5617, hate 0.6674 -> 0.7214, neutral 0.6679 -> 0.7528.
Per-fold macro-F1 baseline [0.5746, 0.6008, 0.5814, 0.5906, 0.5707] vs proposed
[0.6631, 0.7101, 0.6367, 0.6972, 0.6861] - all 5 diffs positive (+0.055 to +0.115).
Wilcoxon signed-rank W=0.0, p=0.0625, not significant at 0.05 - expected: smallest possible
two-sided p with n=5 is 0.0625, so p<0.05 is unattainable. No fold collapsed (~0.19 would mean
single-class prediction; observed range 0.57-0.71 is healthy).



## Rebuild the data from scratch (CPU, local, from Hate speech detection project root)

```
pip install --break-system-packages emoji pandas scikit-learn
python3 1_Data_Preparation/preprocessing.py --input 1_Data_Preparation/annotated_final.csv --outdir 1_Data_Preparation/data_processed
python3 2_Model_Training/make_folds.py --labels_csv 1_Data_Preparation/data_processed/data_with_emoji.csv --outdir 1_Data_Preparation/data_processed --n_splits 5 --seed 42
```

Expect: `common_rows_after_guard 1947`, both CSVs same `comment_id` order, folds
1557/390 and 1558/389 with 0 leaked duplicate texts.

## Verify before submission

- `1947/1947 same order True`, `folds.json` 5 folds, `summary_metrics.csv` 0.5836/0.6786 present
- No fold macro-F1 ~0.19 (collapse rule: retrain that fold for BOTH conditions, state rule in paper)
- `grep -c TODO paper/paper.tex` -> 0 before submit (starts at 26)
- Label audit kappa filled in `DATA_CARD.md` + paper; `OUTDIR` always redefined via Cell 2 after restarts
- Never commit API keys or `.env` files (see `.gitignore`).

## Important notes

- Both variants contain exactly the same comments in the same order (a guard in `preprocessing.py` enforces
  this; `train.py` also asserts it). Without this, `folds.json` would misalign the two conditions.
- With 5 paired folds the smallest possible two-sided Wilcoxon p-value is 0.0625, so p < 0.05 cannot be reached.
  Report this limitation and the per-fold consistency of the effect.
- Hate speech detection project `app.py` uses only inside-Hate speech detection project paths plus verified fallback constants, so the folder is portable.
## Updates
- Added usage instructions