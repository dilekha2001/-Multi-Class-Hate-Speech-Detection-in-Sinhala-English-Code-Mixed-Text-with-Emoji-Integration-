# Multi-Class Hate Speech Detection in Sinhala-English Code-Mixed Text with Emoji Integration (XLM-RoBERTa)

Self-contained runnable project. Copy this folder anywhere and it runs — all paths resolve relative to `app.py`, with verified fallback constants if result files are absent.

**Course:** IT41043 Intelligent Systems, Horizon Campus.
**Research question:** does encoding emoji as text improve the macro-F1 of fine-tuned XLM-RoBERTa on three-class (neutral / offensive / hate) Sinhala-English code-mixed Facebook + YouTube comments?

**Result (verified Colab T4 run, 5 paired folds):** proposed **0.6786 ± 0.0291** vs baseline **0.5836 ± 0.0122** — mean gain **+0.0950**, wins **5/5 folds**, Wilcoxon **W = 0.0, p = 0.0625** (the minimum attainable two-sided p with n = 5, so report the 5/5 consistency, not just p).

---

## Prerequisites

- Python 3.10+ (tested on 3.14, Ubuntu + Colab)
- For the demo only (laptop, no GPU): `streamlit`, `pandas`, `scikit-learn`, `emoji`
- For full training/evaluation: everything in `requirements.txt` (torch, transformers, scipy, …)

```bash
# Demo only
pip install streamlit pandas scikit-learn emoji

# Full project
pip install -r requirements.txt
```

> Ubuntu `externally-managed-environment` (PEP 668): add `--break-system-packages`, or use `pip install --user`, or create a venv (`sudo apt install python3.14-venv` first if needed).

## 60-second demo (laptop, no GPU)

Run from **this folder** (`Hate speech detection project/`):

```bash
python3 -m streamlit run 4_Presentation_and_App/app.py
# open http://localhost:8501
```

What you get — 3 tabs (matching `app.py`):

1. **👉 Live Moderation Tool** — paste a Sinhala-English comment (e.g. `watti amma kenek 😂`), toggle `with_emoji` (proposed: 😡 → `enraged face`) vs `no_emoji` (baseline: emoji stripped), press **Moderate**. The live model is TF-IDF `char_wb(2,5)` + LogisticRegression trained on the same 1,947 comments (notebook Cell 9 recipe, CPU). If the prediction or confidence shifts between toggles, that is the research effect, live. If you re-ran `train.py --save_model`, the best-fold XLM-R checkpoint is used instead.
2. **📊 Experimental Metrics** — static XLM-R 5-fold results: baseline vs proposed macro-F1 cards, per-fold bar chart (5/5 wins), per-class F1 table (offensive is the hardest class), summed confusion-matrix tables, plus a step-by-step "how macro-F1 is counted" explainer. A green banner shows which file the numbers came from; if no result file is found it shows verified fallback constants and tells you which path to add.
3. **⚖️ Statistical Significance** — paired Wilcoxon signed-rank test on the 5 per-fold macro-F1 pairs (W, p, α), per-fold difference table, and why p = 0.0625 is the mathematical floor with n = 5.

## How the app finds its files

All paths in `4_Presentation_and_App/app.py` resolve from `Path(__file__).parent`, so your shell working directory does not matter — only the file layout below matters:

| Needed by | Searched in order (first hit wins) | Status in this repo |
|---|---|---|
| `preprocessing.preprocess_text` | `1_Data_Preparation/preprocessing.py` | ✅ present |
| Demo training data | `1_Data_Preparation/data_processed/data_no_emoji.csv`, `data_with_emoji.csv` (`clean_text`, `label` columns) | ✅ present (1,947 rows each, same order) |
| Headline metrics | `3_Evaluation_and_Stats/outputs_final/summary/summary_metrics.csv` → `3_Evaluation_and_Stats/summary/summary_metrics.csv` → `3_Evaluation_and_Stats/summary_metrics.csv` | ✅ present (last location) → Tab 2 shows **live** numbers |
| Wilcoxon folds | `…/wilcoxon_result.json` (same 3 locations) | ⚠️ absent → Tab 3 folds/W/p from fallback |
| Confusion matrices | `…/summary_metrics_full.json` (same 3 locations) | ⚠️ absent → matrices from fallback |
| XLM-R checkpoint | `…/with_emoji/all_folds_metrics.json` + `with_emoji/fold_N/model/` (`config.json` + weights) | ⚠️ absent (`train.py` ran without `--save_model`) → TF-IDF fallback |

## Folder layout (actual)

```
1_Data_Preparation/
  data_raw/YOUTUBE_FACEBOOK_DATA.xlsx   original data (provenance, do not edit, 2,419 rows)
  annotated_final.csv                   cleaned data (2,210 rows): comment_id, comment, label, source
  DATA_CARD.md                          dataset description (source for the paper's Data section)
  cleaning_report.json                  removals (-3 null, -15 empty, -87 conflicts, -105 dups)
  preprocessing.py                      7-step pipeline + emoji encode/strip toggle + common-row guard
  preprocessing_report.json             drop counts; 99.95% ≤ 128 tokens (justifies max_len=128)
  data_processed/
    data_no_emoji.csv                   baseline input (emoji stripped, 1,947 rows)
    data_with_emoji.csv                 proposed input (emoji → text, 1,947 rows, same order)
    folds.json                          stratified 5-fold split (seed 42), shared by both conditions
2_Model_Training/
  train.py  dataset.py  make_folds.py  label_audit.py
  collection/                           youtube_collect.py, merge_and_scrub.py, annotation_agreement.py
3_Evaluation_and_Stats/
  evaluate.py  stats_test.py            mean ± std + confusion matrices + Wilcoxon
  summary_metrics.csv                   headline table (live source for the app; see table above)
                                        full train also writes outputs_final/{no_emoji,with_emoji}/
                                        + outputs_final/summary/{summary_metrics.csv,
                                          summary_metrics_full.json, wilcoxon_result.json}
4_Presentation_and_App/
  app.py                                Streamlit 3-tab dashboard (this demo)
  paper.tex                             IEEE Access paper source
  Hate_Speech_Detection.pdf             compiled paper
run_full_pipeline.ipynb                 Colab Cells 1–10: train both conditions, evaluate, Wilcoxon, TF-IDF, zip
requirements.txt                        full dependency set
CHECKLIST.md                            done vs to-do
README.md                               this file
.gitignore                              ignores .env, __pycache__/, outputs*/, *.zip, *.bin/safetensors
```

## Results (from `3_Evaluation_and_Stats/summary_metrics.csv`)

| Condition | Macro-F1 | Accuracy | Macro AUC |
|---|---|---|---|
| Baseline (`no_emoji`) | 0.5836 ± 0.0122 | 0.5948 | 0.7673 |
| Proposed (`with_emoji`) | 0.6786 ± 0.0291 | 0.6908 | 0.8526 |

Per-class F1: neutral 0.6679 → 0.7528, offensive 0.4155 → **0.5617** (hardest class, biggest gain), hate 0.6674 → 0.7214.
Per-fold macro-F1 — baseline [0.5746, 0.6008, 0.5814, 0.5906, 0.5707] vs proposed [0.6631, 0.7101, 0.6367, 0.6972, 0.6861]; all 5 diffs positive (+0.055 to +0.115).
Wilcoxon signed-rank W = 0.0, p = 0.0625 — not significant at α = 0.05, **as expected**: with n = 5 the smallest possible two-sided p is 0.0625, so p < 0.05 is unattainable. No fold collapsed (≈0.19 would mean single-class prediction; observed 0.57–0.71 is healthy).

## Reproduce

### Rebuild the data from scratch (CPU, from this folder)

```bash
pip install emoji pandas scikit-learn
python3 1_Data_Preparation/preprocessing.py --input 1_Data_Preparation/annotated_final.csv --outdir 1_Data_Preparation/data_processed
python3 2_Model_Training/make_folds.py --labels_csv 1_Data_Preparation/data_processed/data_with_emoji.csv --outdir 1_Data_Preparation/data_processed --n_splits 5 --seed 42
```

Expect: `common_rows_after_guard 1947`, both CSVs in identical `comment_id` order, folds 1557/390 and 1558/389 with 0 leaked duplicate texts.

### Full training (Google Colab, T4 GPU)

1. Zip **this folder**, upload, and open `run_full_pipeline.ipynb`.
2. Run Cells 1–10 one by one top to bottom (single Play each, not Run all): Cell 1 checks GPU, Cell 2 sets an empty Drive `OUTDIR`, Cell 3 extracts the zip, Cell 4 must print `OK - 1947 rows`, Cell 5 installs deps, Cells 6–7 train baseline (~1.5 h) and proposed (~1.5 h), Cell 8 must show `5/5 True` + the table above, Cell 9 is the TF-IDF reference, Cell 10 downloads `outputs_final.zip`.
3. Unzip into `3_Evaluation_and_Stats/outputs_final/` (keeping the `no_emoji/`, `with_emoji/`, `summary/` subfolders), or run locally:
   ```bash
   python3 3_Evaluation_and_Stats/evaluate.py --outdir <same-OUTDIR-as-train> --savedir 3_Evaluation_and_Stats/outputs_final/summary
   python3 3_Evaluation_and_Stats/stats_test.py --outdir <same-OUTDIR-as-train> --savedir 3_Evaluation_and_Stats/outputs_final/summary
   ```

### Verify before submission

- `1947/1947 same order True`, `folds.json` has 5 folds, `summary_metrics.csv` shows 0.5836 / 0.6786
- No fold macro-F1 ≈ 0.19 (collapse rule: retrain that fold for **both** conditions, state the rule in the paper)
- `grep -c TODO 4_Presentation_and_App/paper.tex` → 0 before submit
- Label-audit kappa filled in `DATA_CARD.md` + paper; Drive `OUTDIR` redefined via Cell 2 after every restart
- Never commit API keys or `.env` (see `.gitignore`)

## Important notes

- Both `data_*.csv` variants contain exactly the same comments in the same order (guard in `preprocessing.py`, asserted again in `train.py`). Without this, `folds.json` would misalign the two conditions.
- With 5 paired folds the smallest possible two-sided Wilcoxon p is 0.0625 — report this limitation and the 5/5 per-fold consistency; do not misread a non-significant p as "no effect".
- `app.py` never leaves this folder except for the embedded fallback constants, so the folder is portable.
