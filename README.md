# Brand Sentiment (Reddit + PySpark + Streamlit)

Local, Parquet-backed brand monitoring with simple scripts:

- Config: edit `config/brands.yml` to set 3 fixed brands, keywords, events, and subreddits.
- Backfill (optional): fetch "top of year" posts from key subreddits to `data/ingest/`.
- Batch process: clean + dedupe + tag brands/events + VADER sentiment; write daily aggregates and monthly top posts to Parquet.
- Dashboard: Streamlit app to visualize volume and sentiment (with 7-day rolling), annotate spikes with event keywords, and browse clickable top posts.
- Live trickle (optional): lightweight producer streams r/all and app can be refreshed to include new files.

## Setup (Windows)

1. Create env and install dependencies:

```cmd
conda create -n brand-sentiment python=3.10 -y
conda activate brand-sentiment
conda install -c conda-forge openjdk=11 -y
python -m pip install -r requirements.txt
```

2. Reddit API (optional, for backfill/stream): create `.env`:

```
REDDIT_CLIENT_ID=your_client_id
REDDIT_CLIENT_SECRET=your_client_secret
REDDIT_USER_AGENT=brand-sentiment-minor/0.1 by u_your_username
```

3. Configure brands: edit `config/brands.yml` (three brands defined by default: Apple, Samsung, Nike).

## Generate dataset (offline-friendly)

- If you already have JSON in `data/ingest/` (from earlier runs), go straight to batch processing.
- Optional backfill (requires `.env`):

```cmd
python backfill_reddit.py --limit 200
```

## Batch process to Parquet

```cmd
python batch_process.py --config config\brands.yml --input data\ingest --out data\parquet --year 2025
```

Outputs:

- `data/parquet/posts/` – cleaned, tagged posts (partitioned by year/month/day/brand)
- `data/parquet/aggregates_daily/` – brand/day metrics
- `data/parquet/top_posts_monthly/` – top-20 posts per brand-month

## Run the dashboard

```cmd
streamlit run streamlit_app.py
```

Use the sidebar to refresh data after new batch runs. Select brand/year, toggle 7-day rolling averages, and click links to open posts.

## Optional live stream (lightweight)

If you want a trickle of live data in `data/ingest/`:

```cmd
python reddit_stream_producer.py --brand "nike,nike inc,air jordan" --flush-interval 5
```

You can refresh the Streamlit app to include new files at any time.

## Notes

- Prefer Parquet for reliability and speed; no infra required beyond local Spark and Python.
- VADER sentiment uses `vaderSentiment` (no extra models). On first run, nothing needs to be downloaded.
- Windows tips:
  - We set `PYSPARK_PYTHON` automatically so PySpark uses your current interpreter.
  - If you run into a "Python worker exited unexpectedly" Spark error, make sure you're executing with the Conda env you created (Python 3.10–3.12). Either `conda activate brand-sentiment` first, or call the interpreter directly, e.g. `"C:\\Users\\mohit\\miniconda3\\envs\\brand-sentiment\\python.exe" batch_process.py ...`.
