# Brand Sentiment Stream (Reddit + PySpark)

Near real-time brand sentiment using Reddit mentions:

- Producer: PRAW streams submissions/comments from r/all, filters by brand, writes NDJSON to `data/ingest/`.
- Consumer: PySpark Structured Streaming reads files, scores with VADER (vaderSentiment), aggregates sentiment per brand over time windows.

## Setup

1. Create Conda env and install deps:

```bash
conda create -n brand-sentiment python=3.10 -y
conda activate brand-sentiment
conda install -c conda-forge openjdk=11 -y
python -m pip install -r requirements.txt
```

2. Create `.env` from the example:

```
REDDIT_CLIENT_ID=your_client_id
REDDIT_CLIENT_SECRET=your_client_secret
REDDIT_USER_AGENT=brand-sentiment-minor/0.1 by u_your_username
```

## Run

Terminal 1 (producer):

```bash
python reddit_stream_producer.py --brand "nike,nike inc,air jordan" --flush-interval 5
```

Terminal 2 (consumer):

```bash
python spark_sentiment_stream.py
```

## Notes

- Adjust keywords via `--brand`.
- Output is printed as rolling 1-minute aggregates every ~10s.
- Data and checkpoints are ignored by Git via `.gitignore`.
