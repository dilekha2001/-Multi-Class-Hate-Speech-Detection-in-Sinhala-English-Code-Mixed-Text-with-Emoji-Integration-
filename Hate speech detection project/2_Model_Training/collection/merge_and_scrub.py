"""
merge_and_scrub.py

Merges the raw YouTube export (youtube_collect.py) with your Facepager
Facebook export (see FACEBOOK_FACEPAGER_SETUP.md) into one raw corpus,
applies the Section 1.3 privacy-stripping step (remove usernames, profile
URLs), and removes exact-duplicate rows BEFORE annotation -- this is the
step that prevents the leakage problems we hit with a prior dataset.

Output is unlabelled -- hand this file to your 3 annotators next
(Section 1.4). It is NOT ready for training until annotation is complete.

Usage:
    python merge_and_scrub.py \
        --youtube ../../data/raw_youtube_comments.csv \
        --facebook ../../data/raw_facebook_comments.csv \
        --outfile ../../data/raw_merged_for_annotation.csv
"""

import argparse
import re

import pandas as pd

URL_RE = re.compile(r"https?://\S+|www\.\S+")
MENTION_RE = re.compile(r"@\w+")
# crude Facebook-profile-link pattern in case raw exports include full URLs
FB_PROFILE_RE = re.compile(r"facebook\.com/[\w.\-]+")


def strip_identifiers(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = URL_RE.sub(" ", text)
    text = FB_PROFILE_RE.sub(" ", text)
    text = MENTION_RE.sub(" ", text)
    return text.strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--youtube", default=None, help="Path to raw YouTube CSV")
    ap.add_argument("--facebook", default=None, help="Path to raw Facebook (Facepager) CSV")
    ap.add_argument("--outfile", required=True)
    args = ap.parse_args()

    if not args.youtube and not args.facebook:
        raise SystemExit("Provide at least one of --youtube or --facebook")

    frames = []
    if args.youtube:
        yt = pd.read_csv(args.youtube, encoding="utf-8-sig")
        yt["source"] = "YouTube"
        frames.append(yt[["comment_id", "comment", "label", "source"]])
        print(f"Loaded {len(yt)} raw YouTube rows.")

    if args.facebook:
        fb = pd.read_csv(args.facebook, encoding="utf-8-sig")
        fb["source"] = "Facebook"
        # adjust these column names to match whatever Facepager actually exported
        if "message" in fb.columns and "comment" not in fb.columns:
            fb = fb.rename(columns={"message": "comment"})
        if "id" in fb.columns and "comment_id" not in fb.columns:
            fb = fb.rename(columns={"id": "comment_id"})
        if "label" not in fb.columns:
            fb["label"] = ""
        frames.append(fb[["comment_id", "comment", "label", "source"]])
        print(f"Loaded {len(fb)} raw Facebook rows.")

    df = pd.concat(frames, ignore_index=True)
    before = len(df)

    # Section 1.3: strip usernames, profile URLs, mentions before storage
    df["comment"] = df["comment"].apply(strip_identifiers)

    # drop empty comments left after scrubbing
    df = df[df["comment"].str.strip().astype(bool)]

    # exact-duplicate check BEFORE annotation, so annotators never see or
    # label the same comment twice, and no duplicate can later leak across
    # train/test folds
    dupes = df.duplicated(subset=["comment"]).sum()
    df = df.drop_duplicates(subset=["comment"]).reset_index(drop=True)

    after = len(df)
    print(f"\nBefore scrub/dedupe: {before} rows")
    print(f"Exact duplicate comments removed: {dupes}")
    print(f"After scrub/dedupe: {after} rows")
    print(f"Source split: {df['source'].value_counts().to_dict()}")

    df.to_csv(args.outfile, index=False, encoding="utf-8-sig")
    print(f"\nWrote {args.outfile}")
    print("This file is UNANNOTATED. Next: run your 3-annotator labelling "
          "process (Section 1.4), compute Cohen's Kappa, resolve disagreements, "
          "THEN feed the labelled result into preprocessing.py.")


if __name__ == "__main__":
    main()
