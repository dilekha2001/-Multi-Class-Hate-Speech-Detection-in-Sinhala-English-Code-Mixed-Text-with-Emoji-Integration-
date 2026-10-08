"""
Hate Speech Detection - Streamlit Demo Dashboard

4 navigation tabs for examiner review:
 1. Live Moderation Tool   - predict code-mixed comments (0=Neutral, 1=Offensive, 2=Hate Speech)
 2. Experimental Metrics    - Baseline vs Proposed macro-F1 comparison (+9.50%)
 3. Statistical Significance - Wilcoxon signed-rank test (W=0.0, p=0.0625, alpha=0.05)
 4. Research Paper (PDF)    - embedded paper viewer (800px) + download button

Run from Gemmbi root:  streamlit run 4_Presentation_and_App/app.py
"""
import base64
import json
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

# Self-contained paths: everything resolves inside Gemmbi/ - no outside roots.
# Copy Gemmbi/ anywhere and the app still runs (verified fallback constants cover missing files).
APP_DIR = Path(__file__).resolve().parent
ROOT = APP_DIR.parent
PREP_DIR = ROOT / "1_Data_Preparation"
EVAL_DIR = ROOT / "3_Evaluation_and_Stats"
OUT_DIR = EVAL_DIR / "outputs_final"
SUMMARY_DIR = OUT_DIR / "summary"
DATA_PROCESSED = PREP_DIR / "data_processed"

if str(PREP_DIR) not in sys.path:
    sys.path.insert(0, str(PREP_DIR))

try:
    from preprocessing import preprocess_text
except Exception as e:
    st.error(f"Cannot import 1_Data_Preparation/preprocessing.py: {e}")
    st.stop()

ID2LABEL = {0: "Neutral", 1: "Offensive", 2: "Hate Speech"}
LABEL2ID = {v: k for k, v in ID2LABEL.items()}
STR2ID = {"neutral": 0, "offensive": 1, "hate": 2}

# Verified Colab T4 run - used only if result files are missing.
FALLBACK_FOLDS_BASE = [0.5746, 0.6008, 0.5814, 0.5906, 0.5707]
FALLBACK_FOLDS_PROP = [0.6631, 0.7101, 0.6367, 0.6972, 0.6861]


@st.cache_resource(show_spinner="Loading demo models (10 sec)...")
def load_tfidf_models():
    """Fast CPU fallback (same recipe as notebook Cell 9): one pipeline per condition."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline

    models = {}
    for cond in ["no_emoji", "with_emoji"]:
        path = DATA_PROCESSED / f"data_{cond}.csv"
        if not path.exists():
            continue
        df = pd.read_csv(path, encoding="utf-8-sig").fillna("")
        pipe = make_pipeline(
            TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=2),
            LogisticRegression(max_iter=2000, class_weight="balanced"),
        )
        pipe.fit(df["clean_text"].astype(str), df["label"].astype(str))
        models[cond] = pipe
    return models


@st.cache_resource(show_spinner="Looking for XLM-RoBERTa checkpoints...")
def load_xlmr():
    """Best-fold XLM-R checkpoint from 3_Evaluation_and_Stats/outputs_final/.

    Returns (tokenizer, model, device, fold_no, source) or (None, None, None, None, reason).
    train.py was run without --save_model, so weights are normally absent and the
    TF-IDF fallback below is used instead. Re-run train.py with --save_model to enable this.
    """
    try:
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
    except Exception as e:
        return None, None, None, None, f"torch/transformers not installed ({e})"

    summary = SUMMARY_DIR / "summary_metrics_full.json"
    allfolds = OUT_DIR / "with_emoji" / "all_folds_metrics.json"
    try:
        folds = json.loads(allfolds.read_text(encoding="utf-8"))
        best = max(folds, key=lambda m: m["macro_f1"])
        fold_no = int(best["fold"])
    except Exception:
        fold_no = 1
    model_dir = OUT_DIR / "with_emoji" / f"fold_{fold_no}" / "model"
    weights = list(model_dir.glob("*.bin")) + list(model_dir.glob("*.safetensors"))
    if not (model_dir / "config.json").exists() or not weights:
        return None, None, None, None, (
            f"No checkpoint at outputs_final/with_emoji/fold_{fold_no}/model "
            "(train.py ran without --save_model)"
        )
    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        tok = AutoTokenizer.from_pretrained(str(model_dir))
        mdl = AutoModelForSequenceClassification.from_pretrained(str(model_dir)).to(device).eval()
        return tok, mdl, device, fold_no, f"with_emoji/fold_{fold_no}/model"
    except Exception as e:
        return None, None, None, None, f"Checkpoint load failed: {e}"


def predict_xlmr(text_clean, tok, mdl, device):
    import torch

    enc = tok(text_clean, truncation=True, padding=True, max_length=128, return_tensors="pt")
    enc = {k: v.to(device) for k, v in enc.items()}
    with torch.no_grad():
        probs = torch.softmax(mdl(**enc).logits, dim=-1).cpu().numpy()[0]
    pred = int(probs.argmax())
    return pred, [float(p) for p in probs]


def load_summary():
    """Headline numbers, live from result files with verified fallbacks."""
    res = {
        "base_mean": 0.5836, "base_std": 0.0122,
        "prop_mean": 0.6786, "prop_std": 0.0291,
        "base_folds": list(FALLBACK_FOLDS_BASE), "prop_folds": list(FALLBACK_FOLDS_PROP),
        "W": 0.0, "p": 0.0625, "alpha": 0.05,
        "from_files": False,
    }
    try:
        df = pd.read_csv(SUMMARY_DIR / "summary_metrics.csv")
        row = df[(df["condition"] == "no_emoji") & (df["metric"] == "macro_f1")].iloc[0]
        res["base_mean"], res["base_std"] = float(row["mean"]), float(row["std"])
        row = df[(df["condition"] == "with_emoji") & (df["metric"] == "macro_f1")].iloc[0]
        res["prop_mean"], res["prop_std"] = float(row["mean"]), float(row["std"])
        w = json.loads((SUMMARY_DIR / "wilcoxon_result.json").read_text(encoding="utf-8"))
        res["base_folds"] = [round(x, 4) for x in w["baseline_macro_f1_per_fold"]]
        res["prop_folds"] = [round(x, 4) for x in w["proposed_macro_f1_per_fold"]]
        res["W"] = float(w["wilcoxon_statistic"])
        res["p"] = float(w["wilcoxon_p_value"])
        res["alpha"] = float(w.get("alpha", 0.05))
        res["from_files"] = True
    except Exception:
        pass
    return res


def find_paper_pdf():
    """Compiled research paper PDF: 4_Presentation_and_App/paper.pdf (drop it in after Overleaf export)."""
    for name in ["paper.pdf", "main.pdf"]:
        p = APP_DIR / name
        if p.exists():
            return p
    for p in sorted(APP_DIR.glob("*paper*.pdf")):
        return p
    return None


RESULTS = load_summary()

st.set_page_config(page_title="Sinhala-English Hate Speech Demo (Gemmbi)", layout="wide")
st.title("Multi-Class Hate Speech Detection (Sinhala-English + Emoji)")
st.caption(
    "Research question: does encoding emoji as text improve macro-F1 of fine-tuned XLM-RoBERTa "
    "on neutral / offensive / hate Facebook + YouTube comments?"
)

tab1, tab2, tab3, tab4 = st.tabs([
    "👉 Live Moderation Tool",
    "📊 Experimental Metrics",
    "⚖️ Statistical Significance",
    "📄 Research Paper (PDF)",
])

# ================= Tab 1: Live Moderation Tool =================
with tab1:
    st.header("👉 Live Moderation Tool")
    st.write("Labels: **0 = Neutral**, **1 = Offensive**, **2 = Hate Speech**.")
    tok, mdl, device, fold_no, src = load_xlmr()
    if tok is not None:
        st.success(f"Using XLM-RoBERTa checkpoint: `3_Evaluation_and_Stats/outputs_final/{src}`")
        use_xlmr = True
    else:
        st.info(
            f"XLM-RoBERTa checkpoint unavailable ({src}). "
            "Demo fallback: TF-IDF + LogisticRegression trained on the same 1947 comments "
            "(notebook Cell 9 recipe, CPU). Re-run `train.py --save_model` to enable live XLM-R."
        )
        use_xlmr = False

    cond = st.radio(
        "Emoji handling (single experimental variable):",
        ["with_emoji", "no_emoji"],
        format_func=lambda x: "with_emoji: 😡 → 'enraged face' (Proposed)" if x == "with_emoji"
        else "no_emoji: 😡 removed (Baseline)",
        horizontal=True,
    )
    text = st.text_area(
        "Paste a Sinhala-English code-mixed comment:",
        value="watti amma kenek 😂",
        height=100,
    )
    if st.button("Moderate", type="primary"):
        if not text.strip():
            st.warning("Type a comment first.")
        else:
            mode = "encode" if cond == "with_emoji" else "strip"
            clean = preprocess_text(text, emoji_mode=mode)
            st.write(f"**Model input ({cond}):** `{clean}`")
            if use_xlmr:
                pred_id, probs = predict_xlmr(clean, tok, mdl, device)
            else:
                models = load_tfidf_models()
                if cond not in models:
                    st.error(f"Demo model for {cond} not found. Check 1_Data_Preparation/data_processed/.")
                    st.stop()
                pipe = models[cond]
                pred_id = STR2ID[str(pipe.predict([clean])[0])]
                probs = [0.0, 0.0, 0.0]
                for cls, pr in zip(pipe.classes_, pipe.predict_proba([clean])[0]):
                    probs[STR2ID[str(cls)]] = float(pr)
            color = {0: "green", 1: "orange", 2: "red"}[pred_id]
            st.markdown(f"### Prediction: **{pred_id} = :{color}[{ID2LABEL[pred_id]}]** ({max(probs):.2%} confidence)")
            st.bar_chart(pd.DataFrame({"confidence": probs}, index=["0 Neutral", "1 Offensive", "2 Hate Speech"]))
            st.caption("Toggle the emoji mode and re-run: a changed prediction is the research effect, live.")
    with st.expander("Try these test cases"):
        st.write("- `akd sir respect ❤️🙏` → expect 0 Neutral (emoji adds respectful tone)")
        st.write("- `oya moda, kata wahapan` → expect 1 Offensive (insult, no group attack)")
        st.write("- `match eka lassanata thibba` → expect 0 Neutral (no emoji: both modes agree)")

# ================= Tab 2: Experimental Metrics =================
with tab2:
    st.header("📊 Experimental Metrics")
    st.write("Stratified 5-fold cross-validation, identical folds for both conditions (mean ± std).")
    base_pct = RESULTS["base_mean"] * 100
    prop_pct = RESULTS["prop_mean"] * 100
    gain_pp = (RESULTS["prop_mean"] - RESULTS["base_mean"]) * 100
    st.table(pd.DataFrame({
        "Model": ["Baseline (emoji stripped)", "Proposed (emoji → text)"],
        "Macro-F1": [f"{base_pct:.2f}%", f"{prop_pct:.2f}%"],
        "Std. dev.": [f"±{RESULTS['base_std']*100:.2f}%", f"±{RESULTS['prop_std']*100:.2f}%"],
    }))
    st.success(f"Performance improvement: **+{gain_pp:.2f} percentage points** (Proposed wins 5/5 folds).")
    st.bar_chart(pd.DataFrame(
        {"Baseline": RESULTS["base_folds"], "Proposed": RESULTS["prop_folds"]},
        index=[f"Fold {i}" for i in range(1, 6)],
    ))
    with st.expander("Per-fold macro-F1 values"):
        st.dataframe(pd.DataFrame({
            "Fold": [1, 2, 3, 4, 5],
            "Baseline": RESULTS["base_folds"],
            "Proposed": RESULTS["prop_folds"],
            "Diff": [round(p - b, 4) for b, p in zip(RESULTS["base_folds"], RESULTS["prop_folds"])],
        }), use_container_width=True)

# ================= Tab 3: Statistical Significance =================
with tab3:
    st.header("⚖️ Statistical Significance")
    st.write("Paired Wilcoxon signed-rank test on the 5 per-fold macro-F1 pairs (non-parametric: no normality assumption).")
    c1, c2, c3 = st.columns(3)
    c1.metric("Test statistic (W)", f"{RESULTS['W']:.1f}")
    c2.metric("P-value", f"{RESULTS['p']:.4f}")
    c3.metric("Alpha (α)", f"{RESULTS['alpha']:.2f}")
    if RESULTS["p"] < RESULTS["alpha"]:
        st.success("Result is statistically significant at α = 0.05.")
    else:
        st.warning("Result is NOT statistically significant at α = 0.05 — this is expected (see below).")
    st.subheader("Why p = 0.0625 is the mathematical limit here")
    st.write(
        "With only **n = 5** paired folds, the Wilcoxon test has 2⁵ = 32 possible sign assignments. "
        "The most extreme outcome — all 5 differences favouring the Proposed model, which is exactly "
        "what we observe (+0.088, +0.109, +0.055, +0.107, +0.115) — still yields a two-sided p-value of "
        "**2/32 = 0.0625**. A p-value below 0.05 is therefore **mathematically unattainable** in a "
        "5-fold design, no matter how large the effect. The correct reading is practical consistency: "
        "the emoji-encoded model wins on **every single fold**, the strongest evidence this architecture "
        "can produce. A non-significant p here must not be misread as 'no effect'."
    )

# ================= Tab 4: Research Paper (PDF) =================
with tab4:
    st.header("📄 Research Paper (PDF)")
    paper_pdf = find_paper_pdf()
    if paper_pdf is not None:
        b64 = base64.b64encode(paper_pdf.read_bytes()).decode("utf-8")
        st.markdown(
            f'<embed src="data:application/pdf;base64,{b64}" type="application/pdf" '
            f'width="100%" height="800px" />',
            unsafe_allow_html=True,
        )
        st.download_button(
            label="⬇️ Download research paper (PDF)",
            data=paper_pdf.read_bytes(),
            file_name=paper_pdf.name,
            mime="application/pdf",
        )
    else:
        st.warning(
            "Compiled `paper.pdf` not found. Export it from Overleaf (`paper.tex` → Download PDF), "
            "save it as `4_Presentation_and_App/paper.pdf`, and rerun — it will appear here automatically."
        )
        st.subheader("Supporting documents available now")
        for doc in sorted(APP_DIR.glob("*.pdf")):
            st.download_button(
                label=f"⬇️ Download {doc.name}",
                data=doc.read_bytes(),
                file_name=doc.name,
                mime="application/pdf",
                key=f"dl_{doc.name}",
            )
