import os
import sys
import re
import argparse
from datetime import datetime
from typing import List, Dict

# Ensure Spark uses this Python (esp. on Windows)
os.environ.setdefault("PYSPARK_PYTHON", sys.executable)
os.environ.setdefault("PYSPARK_DRIVER_PYTHON", sys.executable)

import yaml
from pyspark.sql import SparkSession, Window
from pyspark.sql.types import (
    StructType, StructField, StringType, LongType, IntegerType, DoubleType, ArrayType, BooleanType
)
from pyspark.sql.functions import (
    col, from_unixtime, to_timestamp, lower, lit, regexp_replace, when, size,
    udf, explode, array_distinct, array, date_format, to_date, count, avg, sum as _sum, max as _max,
)

# VADER via vaderSentiment (already in requirements)
_analyzer = None

def _init_analyzer():
    global _analyzer
    if _analyzer is None:
        try:
            from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
            _analyzer = SentimentIntensityAnalyzer()
        except Exception:
            _analyzer = None
    return _analyzer

def sentiment_compound(text: str) -> float:
    if not text:
        return 0.0
    try:
        analyzer = _init_analyzer()
        if analyzer is None:
            return 0.0
        return float(analyzer.polarity_scores(text).get("compound", 0.0))
    except Exception:
        return 0.0

# UDFs
from pyspark.sql.types import DoubleType
sentiment_udf = udf(sentiment_compound, DoubleType())


def build_keyword_regex(keywords: List[str]) -> re.Pattern:
    terms = [re.escape(k.strip()) for k in keywords if k and k.strip()]
    if not terms:
        return re.compile(r"^$")  # match nothing
    # allow simple word-boundary like behavior, but still match hashtags/usernames
    pat = r"(?i)(?<!\w)(" + "|".join(terms) + r")(?!\w)"
    return re.compile(pat)


def load_config(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    assert "brands" in cfg and isinstance(cfg["brands"], list) and len(cfg["brands"]) == 3, \
        "Config must define exactly 3 brands"
    return cfg


def tag_brands_events(df, brands_cfg: List[Dict], global_events: List[str], cfg: Dict = None):
    # Build brand-level regex patterns broadcast to executors
    # Compile once in driver; UDF will recompile per executor if needed
    brand_defs = []
    for b in brands_cfg:
        brand_defs.append({
            "name": b.get("name"),
            "canonical": b.get("canonical", b.get("name")).lower(),
            "keywords": b.get("keywords", []),
            "events": b.get("events", []),
        })

    # Prepare lowercase text column for keyword matching
    df2 = df.withColumn("text_lc", lower(col("text")))

    # Build a simple subreddit->brand hint map from config: if subreddit name contains brand canonical
    subreddit_hints = {}
    if cfg is not None:
        try:
            subs = [s.strip().lower() for s in (cfg.get("subreddits", []) or []) if s and s.strip()]
            for b in brand_defs:
                can = b["canonical"].lower()
                for s in subs:
                    if can in s:
                        subreddit_hints.setdefault(s, set()).add(can)
        except Exception:
            pass

    # Generate arrays: matched brands (canonical) and event keywords for that brand
    def match_brands_events(text: str, subreddit: str):
        tl = (text or "").lower()
        sub = (subreddit or "").lower()
        matched_brands = []
        matched_events = []
        # keyword-based brand detection
        for b in brand_defs:
            for kw in b["keywords"]:
                if kw and kw.lower() in tl:
                    matched_brands.append(b["canonical"])
                    break
            # brand-specific events
            evs = []
            for ek in b.get("events", []):
                if ek and ek.lower() in tl:
                    evs.append(ek.lower())
            if evs:
                matched_events.extend(evs)
        # subreddit hint: if subreddit name contains brand canonical
        if subreddit_hints:
            hint = subreddit_hints.get(sub)
            if hint:
                matched_brands.extend(list(hint))
        # global events
        for gk in global_events or []:
            if gk and gk.lower() in tl:
                matched_events.append(gk.lower())
        # distinct
        return (list(dict.fromkeys(matched_brands)), list(dict.fromkeys(matched_events)))

    from pyspark.sql.types import StructType as S, StructField as F
    from pyspark.sql.types import ArrayType as A

    @udf(S([F("brands", A(StringType())), F("events", A(StringType()))]))
    def match_udf(text: str, subreddit: str):
        b, e = match_brands_events(text, subreddit)
        return {"brands": b, "events": e}

    tagged = df2.withColumn("match", match_udf(col("text"), col("subreddit"))) \
               .withColumn("brands", col("match.brands")) \
               .withColumn("event_tags", array_distinct(col("match.events"))) \
               .drop("match", "text_lc")

    # explode into per-brand rows
    exploded = tagged.where(size(col("brands")) > 0).withColumn("brand", explode(col("brands"))).drop("brands")
    return exploded


def build_spark(app_name: str = "BrandSentimentBatch"):
    spark = (SparkSession.builder
             .appName(app_name)
             .master("local[*]")
             .config("spark.sql.shuffle.partitions", "4")
             .getOrCreate())
    spark.sparkContext.setLogLevel("WARN")
    return spark


def main():
    parser = argparse.ArgumentParser(description="Batch process Reddit data to Parquet aggregates and top posts")
    parser.add_argument("--config", default=os.path.join("config", "brands.yml"))
    parser.add_argument("--input", default=os.path.join("data", "ingest"), help="Input folder with NDJSON files")
    parser.add_argument("--out", default=os.path.join("data", "parquet"), help="Output base folder for Parquet")
    parser.add_argument("--year", default=None, help="Restrict to a specific year (YYYY) for TOY backfill")
    args = parser.parse_args()

    cfg = load_config(args.config)
    brands_cfg = cfg.get("brands", [])
    global_events = cfg.get("global_events", [])

    spark = build_spark()

    # Define schema matching the producer output
    schema = StructType([
        StructField("id", StringType()),
        StructField("type", StringType()),
        StructField("subreddit", StringType()),
        StructField("author", StringType()),
        StructField("created_utc", LongType()),
        StructField("text", StringType()),
        StructField("url", StringType()),
        StructField("score", IntegerType()),
        StructField("num_comments", IntegerType()),
        StructField("parent_id", StringType()),
        StructField("brand", StringType()),  # producer brand (may differ from canonical tagging)
        StructField("fetched_at", LongType()),
        StructField("permalink", StringType()),
        # optional fields (if present from other sources will be ignored)
    ])

    input_path = args.input
    if not os.path.exists(input_path):
        print(f"Input folder not found: {input_path}")
        sys.exit(1)

    df = spark.read.schema(schema).json(input_path)

    # Normalize text: combine title+body already done in producer; just strip weird whitespace
    df = df.withColumn("text", regexp_replace(col("text"), r"\s+", " ").cast(StringType()))

    # Timestamps and date parts
    df = df.withColumn("event_ts", to_timestamp(from_unixtime(col("created_utc")))) \
           .withColumn("event_date", to_date(col("event_ts"))) \
           .withColumn("year", date_format(col("event_date"), "yyyy")) \
           .withColumn("month", date_format(col("event_date"), "MM")) \
           .withColumn("day", date_format(col("event_date"), "dd"))

    if args.year:
        df = df.where(col("year") == lit(args.year))

    # Dedupe by id, keep most recent by created_utc, then fetched_at
    w = Window.partitionBy("id").orderBy(col("created_utc").desc(), col("fetched_at").desc())
    from pyspark.sql.functions import row_number
    df = df.withColumn("rn", row_number().over(w)).where(col("rn") == 1).drop("rn")

    # Brand + event tagging (override/augment producer brand)
    tagged = tag_brands_events(df, brands_cfg, global_events, cfg)

    # Sentiment
    scored = tagged.withColumn("compound", sentiment_udf(col("text"))) \
                   .withColumn("sentiment",
                               when(col("compound") > 0.05, lit("positive"))
                               .when(col("compound") < -0.05, lit("negative"))
                               .otherwise(lit("neutral"))) \
                   .withColumn("event_hit", when(size(col("event_tags")) > 0, lit(True)).otherwise(lit(False)))

    # Persist cleaned posts for dashboard (partitioned by year/month/day)
    posts_out = os.path.join(args.out, "posts")
    (scored
     .repartition("year", "month", "day", "brand")
     .write.mode("overwrite")
     .partitionBy("year", "month", "day", "brand")
     .parquet(posts_out))

    # Daily aggregates per brand
    agg = (scored.groupBy("brand", "event_date")
           .agg(
               count(lit(1)).alias("mentions"),
               avg("compound").alias("avg_compound"),
               _sum((col("sentiment") == "positive").cast("int")).alias("positive"),
               _sum((col("sentiment") == "negative").cast("int")).alias("negative"),
               _sum((col("sentiment") == "neutral").cast("int")).alias("neutral"),
               _sum(col("event_hit").cast("int")).alias("event_hits"),
               _sum((size(col("event_tags")) > 0).cast("int")).alias("event_posts"),
               _max("score").alias("max_score"),
           )
          )

    daily_out = os.path.join(args.out, "aggregates_daily")
    (agg
     .withColumn("year", date_format(col("event_date"), "yyyy"))
     .withColumn("month", date_format(col("event_date"), "MM"))
     .repartition("year", "month", "brand")
     .write.mode("overwrite")
     .partitionBy("year", "month", "brand")
     .parquet(daily_out))

    # Top-20 posts per month per brand by score
    from pyspark.sql.functions import dense_rank
    w2 = Window.partitionBy("brand", "year", "month").orderBy(col("score").desc_nulls_last())
    ranked = (scored
              .withColumn("rank", dense_rank().over(w2))
              .where(col("rank") <= 20)
             )

    tops_out = os.path.join(args.out, "top_posts_monthly")
    (ranked
     .select("brand", "year", "month", "id", "type", "subreddit", "author", "event_ts", "score",
             "num_comments", "url", "permalink", "compound", "sentiment", "event_tags", "text")
     .repartition("year", "month", "brand")
     .write.mode("overwrite")
     .partitionBy("year", "month", "brand")
     .parquet(tops_out))

    print(f"Wrote posts -> {posts_out}")
    print(f"Wrote daily aggregates -> {daily_out}")
    print(f"Wrote monthly top posts -> {tops_out}")

    spark.stop()


if __name__ == "__main__":
    main()
