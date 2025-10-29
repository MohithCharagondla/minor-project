import argparse
import json
import os
import queue
import re
import signal
import sys
import threading
import time
import uuid
from datetime import datetime, timezone

from dotenv import load_dotenv
import praw

INGEST_DIR = os.path.join(os.path.dirname(__file__), "data", "ingest")

def load_reddit():
    load_dotenv()
    client_id = os.getenv("REDDIT_CLIENT_ID")
    client_secret = os.getenv("REDDIT_CLIENT_SECRET")
    user_agent = os.getenv("REDDIT_USER_AGENT", "brand-sentiment-minor/0.1")
    if not (client_id and client_secret and user_agent):
        print("Missing Reddit API credentials. Fill .env (REDDIT_CLIENT_ID, REDDIT_CLIENT_SECRET, REDDIT_USER_AGENT).")
        sys.exit(1)
    reddit = praw.Reddit(
        client_id=client_id,
        client_secret=client_secret,
        user_agent=user_agent,
        check_for_async=False,
    )
    reddit.read_only = True
    return reddit

def compile_brand_regex(keywords):
    esc = [re.escape(k.strip()) for k in keywords if k.strip()]
    # word boundary-ish; allow hashtags/mentions too
    pattern = r"(?i)(?<!\w)(" + "|".join(esc) + r")(?!\w)"
    return re.compile(pattern)

def ensure_dirs():
    os.makedirs(INGEST_DIR, exist_ok=True)

def make_record_from_submission(s, brand):
    text = f"{s.title or ''}\n{s.selftext or ''}".strip()
    return {
        "id": s.id,
        "type": "submission",
        "subreddit": str(s.subreddit),
        "author": str(getattr(s.author, "name", "")) if s.author else "",
        "created_utc": int(s.created_utc),
        "text": text,
        "url": f"https://www.reddit.com{s.permalink}",
        "score": int(getattr(s, "score", 0) or 0),
        "num_comments": int(getattr(s, "num_comments", 0) or 0),
        "parent_id": None,
        "brand": brand,
        "fetched_at": int(time.time()),
        "permalink": f"https://www.reddit.com{s.permalink}",
    }

def make_record_from_comment(c, brand):
    return {
        "id": c.id,
        "type": "comment",
        "subreddit": str(c.subreddit),
        "author": str(getattr(c.author, "name", "")) if c.author else "",
        "created_utc": int(c.created_utc),
        "text": c.body or "",
        "url": f"https://www.reddit.com{c.permalink}",
        "score": int(getattr(c, "score", 0) or 0),
        "num_comments": None,
        "parent_id": c.parent_id,
        "brand": brand,
        "fetched_at": int(time.time()),
        "permalink": f"https://www.reddit.com{c.permalink}",
    }

class BufferedFileWriter:
    def __init__(self, out_dir, flush_interval=5, max_batch=200):
        self.out_dir = out_dir
        self.flush_interval = flush_interval
        self.max_batch = max_batch
        self.q = queue.Queue()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def start(self):
        self._thread.start()

    def stop(self):
        self._stop.set()
        self._thread.join(timeout=5)

    def put(self, item):
        self.q.put(item)

    def _drain(self, max_items):
        items = []
        try:
            while len(items) < max_items:
                items.append(self.q.get_nowait())
        except queue.Empty:
            pass
        return items

    def _run(self):
        while not self._stop.is_set():
            time.sleep(self.flush_interval)
            batch = self._drain(self.max_batch)
            if not batch:
                continue
            ts_ms = int(time.time() * 1000)
            fname = f"reddit_{ts_ms}_{uuid.uuid4().hex}.json"
            tmp = os.path.join(self.out_dir, f".tmp_{fname}")
            final = os.path.join(self.out_dir, fname)
            try:
                with open(tmp, "w", encoding="utf-8") as f:
                    for rec in batch:
                        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                os.replace(tmp, final)  # atomic for Spark
                print(f"Wrote {len(batch)} records -> {final}")
            except Exception as e:
                print(f"Write error: {e}", file=sys.stderr)
                try:
                    if os.path.exists(tmp):
                        os.remove(tmp)
                except Exception:
                    pass

def stream_worker_submissions(reddit, brand_pat, brands_set, writer, seen_ids, seen_lock):
    for s in reddit.subreddit("all").stream.submissions(skip_existing=True):
        try:
            text = f"{s.title or ''}\n{s.selftext or ''}"
            m = brand_pat.search(text)
            if not m:
                continue
            brand = m.group(0).lower()
            if brands_set and brand not in brands_set:
                # When keywords have overlaps, keep only the matched canonical
                pass
            with seen_lock:
                if s.id in seen_ids:
                    continue
                seen_ids.add(s.id)
                if len(seen_ids) > 100_000:
                    # prevent unbounded growth
                    for _ in range(10_000):
                        seen_ids.pop()
            rec = make_record_from_submission(s, brand)
            writer.put(rec)
        except Exception as e:
            print(f"Submissions stream error: {e}", file=sys.stderr)
            time.sleep(1)

def stream_worker_comments(reddit, brand_pat, brands_set, writer, seen_ids, seen_lock):
    for c in reddit.subreddit("all").stream.comments(skip_existing=True):
        try:
            text = c.body or ""
            m = brand_pat.search(text)
            if not m:
                continue
            brand = m.group(0).lower()
            with seen_lock:
                if c.id in seen_ids:
                    continue
                seen_ids.add(c.id)
                if len(seen_ids) > 100_000:
                    for _ in range(10_000):
                        seen_ids.pop()
            rec = make_record_from_comment(c, brand)
            writer.put(rec)
        except Exception as e:
            print(f"Comments stream error: {e}", file=sys.stderr)
            time.sleep(1)

def main():
    parser = argparse.ArgumentParser(description="Reddit brand mention producer -> data/ingest")
    parser.add_argument("--brand", required=True, help="Brand keywords, comma-separated (e.g., nike,nike inc,air jordan)")
    parser.add_argument("--flush-interval", type=int, default=5, help="Seconds between file flushes")
    parser.add_argument("--max-batch", type=int, default=200, help="Max records per file")
    args = parser.parse_args()

    keywords = [k.strip() for k in args.brand.split(",") if k.strip()]
    brand_pat = compile_brand_regex(keywords)
    brands_set = set([k.lower() for k in keywords])

    ensure_dirs()
    reddit = load_reddit()

    writer = BufferedFileWriter(INGEST_DIR, flush_interval=args.flush_interval, max_batch=args.max_batch)
    writer.start()

    # Simple seen-id structure
    seen_ids = set()
    seen_lock = threading.Lock()

    threads = [
        threading.Thread(target=stream_worker_submissions, args=(reddit, brand_pat, brands_set, writer, seen_ids, seen_lock), daemon=True),
        threading.Thread(target=stream_worker_comments, args=(reddit, brand_pat, brands_set, writer, seen_ids, seen_lock), daemon=True),
    ]
    for t in threads:
        t.start()

    print(f"Streaming r/all for keywords: {keywords}")
    print("Press Ctrl+C to stop.")
    # Graceful shutdown
    stop_event = threading.Event()
    def handle_sigint(sig, frame):
        stop_event.set()
    signal.signal(signal.SIGINT, handle_sigint)

    try:
        while not stop_event.is_set():
            time.sleep(0.5)
    finally:
        writer.stop()
        print("Stopped.")

if __name__ == "__main__":
    main()
