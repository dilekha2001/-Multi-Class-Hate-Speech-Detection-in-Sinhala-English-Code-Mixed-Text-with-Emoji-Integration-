"""
youtube_collect.py

Collects public top-level comments (and optionally replies) from
YouTube videos via the YouTube Data API v3, matching Section 1.1
of the Milestone 2 report: keyword-searched videos on Sri Lankan
news/political-analysis channels, code-mixed Sinhala-English comments.

Output schema matches your existing dataset so it plugs straight into
preprocessing.py:
    comment_id, comment, label, source, video_id, video_title,
    published_at, like_count

`label` is left BLANK here -- this script only collects raw data.
Annotation (Section 1.4, 3 annotators + Cohen's Kappa) happens
afterward, separately, on the raw collected text.

-----------------------------------------------------------------
SETUP (one-time):
1. Go to https://console.cloud.google.com/
2. Create a project (or use an existing one).
3. APIs & Services -> Library -> enable "YouTube Data API v3".
4. APIs & Services -> Credentials -> Create credentials -> API key.
5. Copy that key and either:
     a) set it as an environment variable:  export YOUTUBE_API_KEY=xxxx
     b) or pass it directly with --api_key xxxx
   Do NOT commit your API key to GitHub. Add a .env or credentials
   file to .gitignore.

QUOTA NOTE: The YouTube Data API free tier gives 10,000 quota units/day.
A search.list call costs 100 units; commentThreads.list costs 1 unit
per page (~100 comments per page). Budget your --max_videos and
--max_comments_per_video accordingly, e.g. searching 10 keywords x
5 videos each = 5,000 units just for search, leaving ~50 comment pages.
Prefer passing explicit --video_ids over --search_query once you've
identified good videos, to save quota.
-----------------------------------------------------------------

Usage (search mode):
    python youtube_collect.py \
        --api_key YOUR_KEY \
        --search_query "Sri Lanka election news" \
        --max_videos 5 \
        --max_comments_per_video 200 \
        --outfile ../../data/raw_youtube_comments.csv

Usage (explicit video IDs, saves quota):
    python youtube_collect.py \
        --api_key YOUR_KEY \
        --video_ids dQw4w9WgXcQ,abc123XYZ \
        --max_comments_per_video 300 \
        --outfile ../../data/raw_youtube_comments.csv
"""

import argparse
import csv
import os
import sys
import time

from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

API_SERVICE_NAME = "youtube"
API_VERSION = "v3"


def search_videos(youtube, query, max_videos, region_code="LK"):
    """Find video IDs via keyword search, biased to Sri Lanka region."""
    video_ids = []
    request = youtube.search().list(
        q=query,
        part="id,snippet",
        type="video",
        maxResults=min(max_videos, 50),
        regionCode=region_code,
        relevanceLanguage="si",  # Sinhala relevance hint; comments are still code-mixed
    )
    response = request.execute()
    for item in response.get("items", []):
        video_ids.append({
            "video_id": item["id"]["videoId"],
            "video_title": item["snippet"]["title"],
        })
    return video_ids[:max_videos]


def fetch_comments_for_video(youtube, video_id, video_title, max_comments):
    """Paginate commentThreads.list for one video."""
    rows = []
    next_page_token = None
    fetched = 0

    while fetched < max_comments:
        try:
            request = youtube.commentThreads().list(
                part="snippet",
                videoId=video_id,
                maxResults=min(100, max_comments - fetched),
                pageToken=next_page_token,
                textFormat="plainText",
                order="relevance",  # surfaces high-engagement comments first
            )
            response = request.execute()
        except HttpError as e:
            if e.resp.status == 403:
                print(f"  Comments disabled or forbidden for video {video_id}, skipping.")
            else:
                print(f"  API error on video {video_id}: {e}")
            break

        for item in response.get("items", []):
            top = item["snippet"]["topLevelComment"]["snippet"]
            rows.append({
                "comment_id": item["snippet"]["topLevelComment"]["id"],
                "comment": top["textDisplay"],
                "label": "",  # to be filled during annotation (Section 1.4)
                "source": "YouTube",
                "video_id": video_id,
                "video_title": video_title,
                "published_at": top["publishedAt"],
                "like_count": top.get("likeCount", 0),
            })
            fetched += 1
            if fetched >= max_comments:
                break

        next_page_token = response.get("nextPageToken")
        if not next_page_token:
            break
        time.sleep(0.2)  # be polite to the API

    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--api_key", default=os.environ.get("YOUTUBE_API_KEY"),
                     help="YouTube Data API v3 key. Falls back to YOUTUBE_API_KEY env var.")
    ap.add_argument("--search_query", default=None,
                     help="Keyword search (e.g. 'Sri Lanka election news'). "
                          "Costs 100 quota units per call.")
    ap.add_argument("--video_ids", default=None,
                     help="Comma-separated explicit video IDs, skips search "
                          "(cheaper on quota). e.g. --video_ids abc123,def456")
    ap.add_argument("--max_videos", type=int, default=5,
                     help="Only used with --search_query.")
    ap.add_argument("--max_comments_per_video", type=int, default=200)
    ap.add_argument("--outfile", required=True)
    args = ap.parse_args()

    if not args.api_key:
        sys.exit("No API key provided. Use --api_key or set YOUTUBE_API_KEY env var.")
    if not args.search_query and not args.video_ids:
        sys.exit("Provide either --search_query or --video_ids.")

    youtube = build(API_SERVICE_NAME, API_VERSION, developerKey=args.api_key)

    if args.video_ids:
        videos = [{"video_id": vid.strip(), "video_title": ""} for vid in args.video_ids.split(",")]
    else:
        print(f"Searching for videos matching: '{args.search_query}'")
        videos = search_videos(youtube, args.search_query, args.max_videos)
        print(f"Found {len(videos)} videos.")

    all_rows = []
    for v in videos:
        print(f"Fetching comments for video {v['video_id']} ({v['video_title']})...")
        rows = fetch_comments_for_video(youtube, v["video_id"], v["video_title"],
                                         args.max_comments_per_video)
        print(f"  Got {len(rows)} comments.")
        all_rows.extend(rows)

    if not all_rows:
        print("No comments collected. Check that the videos have comments enabled "
              "and your API key/quota are valid.")
        return

    fieldnames = ["comment_id", "comment", "label", "source", "video_id",
                  "video_title", "published_at", "like_count"]
    os.makedirs(os.path.dirname(args.outfile) or ".", exist_ok=True)
    with open(args.outfile, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)

    print(f"\nWrote {len(all_rows)} raw comments to {args.outfile}")
    print("NOTE: 'label' column is empty -- these are RAW, UNANNOTATED comments. "
          "Run your 3-annotator labelling process (Section 1.4) before using this "
          "for training. Also apply your privacy-stripping step (Section 1.3) to "
          "remove any residual usernames before storage.")


if __name__ == "__main__":
    main()
