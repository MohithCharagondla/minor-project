import os
import sys
import time
import json
import argparse
from datetime import datetime, timezone
from typing import List

from dotenv import load_dotenv

# Output folder
BASE_DIR = os.path.dirname(__file__)
INGEST_DIR = os.path.join(BASE_DIR, "data", "ingest")


def load_reddit():
    load_dotenv()
    try:
        import praw
        import prawcore
    except ImportError:
        print("praw not installed. Please install requirements.txt")
        sys.exit(1)
    client_id = os.getenv("REDDIT_CLIENT_ID")
    client_secret = os.getenv("REDDIT_CLIENT_SECRET")
    user_agent = os.getenv("REDDIT_USER_AGENT", "brand-sentiment-minor/0.1")
    if not (client_id and client_secret and user_agent):
        print("Missing Reddit API credentials in .env; backfill requires API access. Falling back to offline data.")
        return None
    reddit = praw.Reddit(
        client_id=client_id,
        client_secret=client_secret,
        user_agent=user_agent,
        check_for_async=False,
    )
    reddit.read_only = True
    return reddit


def ensure_dirs():
    os.makedirs(INGEST_DIR, exist_ok=True)


def write_ndjson(records: List[dict]):
    if not records:
        return None
    ts_ms = int(time.time() * 1000)
    fname = f"backfill_{ts_ms}.json"
    path = os.path.join(INGEST_DIR, fname)
    with open(path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    return path


def submission_to_record(s):
    text = f"{s.title or ''}\n{s.selftext or ''}".strip()
    return {
        "id": s.id,
        "type": "submission",
        "subreddit": str(s.subreddit),
        "author": str(getattr(s.author, "name", "")) if s.author else "",
        "created_utc": int(getattr(s, "created_utc", time.time())),
        "text": text,
        "url": f"https://www.reddit.com{s.permalink}",
        "score": int(getattr(s, "score", 0) or 0),
        "num_comments": int(getattr(s, "num_comments", 0) or 0),
        "parent_id": None,
        "brand": None,
        "fetched_at": int(time.time()),
        "permalink": f"https://www.reddit.com{s.permalink}",
    }


def backfill_top_of_year(subreddits: List[str], limit_per_sub=200, base_delay=1.0):
    reddit = load_reddit()
    if reddit is None:
        print("Skipping API backfill; no credentials. Use offline dataset in data/ingest.")
        return None
    ensure_dirs()

    records = []
    year = datetime.now(timezone.utc).year
    print(f"Backfilling 'top of year' posts for {year} from: {subreddits}")
    for sub in subreddits:
        try:
            sr = reddit.subreddit(sub)
            # Exponential backoff state
            delay = base_delay
            fetched = 0
            for s in sr.top(time_filter="year", limit=limit_per_sub):
                # Per-item gentle pacing
                time.sleep(0.2)
                try:
                    records.append(submission_to_record(s))
                    fetched += 1
                    # reset backoff on success bursts
                    if fetched % 50 == 0:
                        delay = base_delay
                except Exception as e:
                    # Handle rate limits explicitly
                    try:
                        import prawcore
                        if isinstance(e, prawcore.exceptions.TooManyRequests):
                            sleep_for = getattr(e, 'sleep_time', None) or delay
                            print(f"Rate limited on r/{sub}. Sleeping {sleep_for:.1f}s...")
                            time.sleep(sleep_for)
                            delay = min(delay * 2, 60)
                            continue
                    except Exception:
                        pass
                    # Generic transient error: brief backoff
                    time.sleep(delay)
                    delay = min(delay * 2, 60)
                    continue
            # pacing between subreddits
            time.sleep(1)
        except Exception as e:
            print(f"Subreddit {sub} error: {e}")
            time.sleep(1)
    out = write_ndjson(records)
    if out:
        print(f"Wrote backfill file -> {out} with {len(records)} posts")
    else:
        print("No records written.")
    return out


def load_subreddits_from_config(cfg_path: str) -> List[str]:
    import yaml
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    subs = cfg.get("subreddits", [])
    # ensure unique & clean
    subs = [s.strip() for s in subs if s and s.strip()]
    if not subs:
        subs = ["all"]
    return subs


def main():
    parser = argparse.ArgumentParser(description="Backfill Reddit 'top of year' posts to data/ingest")
    parser.add_argument("--config", default=os.path.join("config", "brands.yml"))
    parser.add_argument("--limit", type=int, default=200, help="Limit per subreddit")
    args = parser.parse_args()

    subs = load_subreddits_from_config(args.config)
    backfill_top_of_year(subs, limit_per_sub=args.limit)


if __name__ == "__main__":
    main()
