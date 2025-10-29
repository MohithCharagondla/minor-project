import os
import sys
from pyspark.sql import SparkSession
from pyspark.sql.types import (
    StructType, StructField, StringType, LongType, IntegerType, DoubleType
)
from pyspark.sql.functions import (
    col, from_unixtime, to_timestamp, udf, when, window, avg, count
)

# Ensure Spark uses this Python (avoids "Missing python3" warning on Windows)
os.environ.setdefault("PYSPARK_PYTHON", sys.executable)
os.environ.setdefault("PYSPARK_DRIVER_PYTHON", sys.executable)

# Lazy init of vaderSentiment analyzer only
_analyzer = None
def _init_analyzer():
    global _analyzer
    if _analyzer is None:
        try:
            from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
            _analyzer = SentimentIntensityAnalyzer()
        except Exception:
            print("Warning: vaderSentiment unavailable; defaulting to neutral scores.")
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

def main():
    base_dir = os.path.dirname(__file__)
    input_dir = os.path.join(base_dir, "data", "ingest")
    checkpoint_dir = os.path.join(base_dir, "data", "checkpoints", "sentiment_console")
    os.makedirs(checkpoint_dir, exist_ok=True)

    spark = (SparkSession.builder
             .appName("BrandSentimentStream")
             .master("local[*]")
             .config("spark.sql.shuffle.partitions", "4")
             .config("spark.sql.streaming.forceDeleteTempCheckpointLocation", "true")
             .getOrCreate())
    spark.sparkContext.setLogLevel("WARN")

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
        StructField("brand", StringType()),
        StructField("fetched_at", LongType()),
        StructField("permalink", StringType()),
    ])

    df = (spark.readStream
          .schema(schema)
          .json(input_dir))

    df = df.withColumn("event_ts", to_timestamp(from_unixtime(col("created_utc"))))

    compound_udf = udf(sentiment_compound, DoubleType())

    scored = (df
              .withColumn("compound", compound_udf(col("text")))
              .withColumn(
                  "sentiment",
                  when(col("compound") > 0.05, "positive")
                  .when(col("compound") < -0.05, "negative")
                  .otherwise("neutral")
              ))

    # 1-minute tumbling window per brand with counts and avg compound
    agg = (scored
           .withWatermark("event_ts", "2 minutes")
           .groupBy(
               window(col("event_ts"), "1 minute").alias("w"),
               col("brand"),
               col("sentiment"),
           )
           .agg(
               count("*").alias("mentions"),
               avg("compound").alias("avg_compound")
           )
           .select(
               col("w.start").alias("window_start"),
               col("w.end").alias("window_end"),
               "brand",
               "sentiment",
               "mentions",
               "avg_compound"
           )
          )

    try:
        query = (agg.writeStream
                 .outputMode("update")
                 .format("console")
                 .option("truncate", False)
                 .option("checkpointLocation", checkpoint_dir)
                 .trigger(processingTime="10 seconds")
                 .start())
        print("Streaming sentiment aggregates to console every ~10s. Press Ctrl+C to stop.")
        query.awaitTermination()
    except KeyboardInterrupt:
        print("Stopping stream...")
    finally:
        try:
            query.stop()
        except Exception:
            pass
        spark.stop()

if __name__ == "__main__":
    main()
