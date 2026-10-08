"""
preprocessing.py

Implements the preprocessing pipeline exactly as specified in
Section 1.5 (Preprocessing Steps) of the Milestone 2 report:

    1. URL removal
    2. Username / mention removal (@user)
    3. Emoji extraction and encoding (Unicode -> text description)
    4. Lowercasing (English tokens only)
    5. Repeated character normalisation (max 2 consecutive chars)
    6. Whitespace normalisation
    7. Minimum length filter (< 3 tokens after preprocessing discarded)

The emoji step is implemented as a TOGGLE (`emoji_mode`), because emoji
handling is the single experimental variable that distinguishes the
baseline model from the proposed model (Section 3, Baseline Definition):

    emoji_mode="encode"  -> proposed model input  (emoji -> "angry face")
    emoji_mode="strip"   -> baseline model input   (emoji removed)

Usage:
    python preprocessing.py \
        --input ../data/synthetic_sinhala_english_2500_comments.csv \
        --outdir ../data

Produces:
    data_with_emoji.csv   (proposed model input)
    data_no_emoji.csv     (baseline model input)
    preprocessing_report.json  (row counts, drop counts -> paste into your report)
"""

import argparse
import json
import re
import sys
from pathlib import Path

import emoji
import pandas as pd

# ----------------------------------------------------------------------
# Individual pipeline steps (Section 1.5, steps 1-6)
# ----------------------------------------------------------------------

URL_RE = re.compile(r"https?://\S+|www\.\S+")
MENTION_RE = re.compile(r"@\w+")
WHITESPACE_RE = re.compile(r"\s+")
# 3+ repeated characters -> collapse to 2 (e.g. "baaaaad" -> "baad")
REPEAT_CHAR_RE = re.compile(r"(.)\1{2,}")
# crude heuristic for "English token": ASCII a-z/A-Z word
ENGLISH_TOKEN_RE = re.compile(r"^[A-Za-z]+$")


def remove_urls(text: str) -> str:
    """Step 1: URL removal."""
    return URL_RE.sub(" ", text)


def remove_mentions(text: str) -> str:
    """Step 2: Username / mention removal (@user)."""
    return MENTION_RE.sub(" ", text)


def encode_emojis(text: str) -> str:
    """
    Step 3 (proposed model): convert each emoji to its Unicode text
    description and append it inline, e.g. "😡" -> "angry face".
    Matches the method described in Section 1.5 step 3 / Section 2.1.
    """
    demojized = emoji.demojize(text, delimiters=(" ", " "))
    # emoji.demojize gives "_angry_face_" style tokens; clean underscores
    demojized = re.sub(r"_", " ", demojized)
    return demojized


def strip_emojis(text: str) -> str:
    """Step 3 (baseline model): remove emojis entirely, matching the
    Muthuthanthri & Smith [1] approach used as the baseline condition."""
    return emoji.replace_emoji(text, replace=" ")


def lowercase_english_tokens(text: str) -> str:
    """
    Step 4: Lowercase English tokens only. Romanised Sinhala tokens are
    left untouched (they are effectively case-insensitive in practice
    per the report's justification); this avoids corrupting mixed-script
    or proper-noun tokens that are not plain ASCII words.
    """
    tokens = text.split(" ")
    out = []
    for tok in tokens:
        if ENGLISH_TOKEN_RE.match(tok):
            out.append(tok.lower())
        else:
            out.append(tok)
    return " ".join(out)


def normalise_repeated_chars(text: str) -> str:
    """Step 5: Collapse 3+ repeated characters down to 2."""
    return REPEAT_CHAR_RE.sub(r"\1\1", text)


def normalise_whitespace(text: str) -> str:
    """Step 6: Collapse multiple spaces/tabs/newlines to a single space."""
    return WHITESPACE_RE.sub(" ", text).strip()


def token_count(text: str) -> int:
    return len([t for t in text.split(" ") if t])


# ----------------------------------------------------------------------
# Full pipeline
# ----------------------------------------------------------------------

def preprocess_text(text: str, emoji_mode: str) -> str:
    """Run steps 1-6 in order for a single comment."""
    if not isinstance(text, str):
        return ""
    text = remove_urls(text)
    text = remove_mentions(text)

    if emoji_mode == "encode":
        text = encode_emojis(text)
    elif emoji_mode == "strip":
        text = strip_emojis(text)
    else:
        raise ValueError("emoji_mode must be 'encode' or 'strip'")

    text = lowercase_english_tokens(text)
    text = normalise_repeated_chars(text)
    text = normalise_whitespace(text)
    return text


def build_variant(df: pd.DataFrame, emoji_mode: str, min_tokens: int = 3) -> pd.DataFrame:
    """
    Apply the full pipeline to the raw `comment` column and apply the
    Step 7 minimum-length filter. Returns a new dataframe plus drop stats.
    """
    out = df.copy()
    out["clean_text"] = out["comment"].apply(lambda t: preprocess_text(t, emoji_mode))
    out["n_tokens"] = out["clean_text"].apply(token_count)

    before = len(out)
    out = out[out["n_tokens"] >= min_tokens].reset_index(drop=True)
    after = len(out)

    stats = {
        "emoji_mode": emoji_mode,
        "rows_before_min_length_filter": before,
        "rows_after_min_length_filter": after,
        "rows_dropped": before - after,
        "drop_rate_pct": round(100 * (before - after) / before, 2) if before else 0.0,
    }
    return out, stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Path to raw annotated CSV")
    ap.add_argument("--outdir", required=True, help="Directory to write processed CSVs")
    ap.add_argument("--min_tokens", type=int, default=3)
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.input, encoding="utf-8-sig")
    required_cols = {"comment_id", "comment", "label"}
    missing = required_cols - set(df.columns)
    if missing:
        sys.exit(f"Input CSV is missing required columns: {missing}")

    print(f"Loaded {len(df)} rows from {args.input}")
    print(df["label"].value_counts())

    report = {"input_rows": len(df), "label_distribution": df["label"].value_counts().to_dict()}

    # Proposed model input: emoji encoded as text
    with_emoji_df, stats_with = build_variant(df, emoji_mode="encode", min_tokens=args.min_tokens)
    with_emoji_path = outdir / "data_with_emoji.csv"
    with_emoji_df.to_csv(with_emoji_path, index=False, encoding="utf-8-sig")
    report["with_emoji"] = stats_with

    # Baseline model input: emoji stripped
    no_emoji_df, stats_no = build_variant(df, emoji_mode="strip", min_tokens=args.min_tokens)
    no_emoji_path = outdir / "data_no_emoji.csv"
    no_emoji_df.to_csv(no_emoji_path, index=False, encoding="utf-8-sig")
    report["no_emoji"] = stats_no

    # ------------------------------------------------------------------
    # Common-row guard: the baseline and proposed models MUST be trained and
    # tested on exactly the same comments in the same order, otherwise
    # folds.json (row positions) misaligns and the paired comparison is invalid.
    # A comment made only of emoji + 1-2 words can pass the length filter when
    # emoji are encoded but fail when they are stripped, so keep only comments
    # that pass in BOTH variants.
    # ------------------------------------------------------------------
    common_ids = set(with_emoji_df["comment_id"]) & set(no_emoji_df["comment_id"])
    n_with, n_no = len(with_emoji_df), len(no_emoji_df)
    with_emoji_df = with_emoji_df[with_emoji_df["comment_id"].isin(common_ids)].reset_index(drop=True)
    no_emoji_df = no_emoji_df[no_emoji_df["comment_id"].isin(common_ids)].reset_index(drop=True)
    assert (with_emoji_df["comment_id"].values == no_emoji_df["comment_id"].values).all()
    with_emoji_df.to_csv(with_emoji_path, index=False, encoding="utf-8-sig")
    no_emoji_df.to_csv(no_emoji_path, index=False, encoding="utf-8-sig")
    report["common_rows_after_guard"] = len(with_emoji_df)
    report["rows_removed_by_common_row_guard"] = {"with_emoji_only_passed": n_with - len(with_emoji_df),
                                                  "no_emoji_only_passed": n_no - len(no_emoji_df)}

    # % of comments within max_seq_len=128 whitespace tokens (sanity check
    # for Section 2.2 Training Configuration justification)
    pct_under_128 = (with_emoji_df["n_tokens"] <= 128).mean() * 100
    report["pct_rows_le_128_tokens"] = round(pct_under_128, 2)

    report_path = outdir / "preprocessing_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print("\n--- Preprocessing complete ---")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"\nWrote: {with_emoji_path}")
    print(f"Wrote: {no_emoji_path}")
    print(f"Wrote: {report_path}")


if __name__ == "__main__":
    main()
