# utils/features_bronze_table.py
import os
from datetime import datetime
from pyspark.sql import functions as F

def _read_csv(spark, path):
    return (spark.read
            .option("header", True)
            .option("inferSchema", True)
            .csv(path))

def _parse_snapshot(col):
    # Handle both "2023-05-01" and "01-05-2023" (and similar)
    return F.coalesce(
        F.to_date(col),
        F.to_date(col, "yyyy-MM-dd"),
        F.to_date(col, "dd-MM-yyyy"),
        F.to_date(col, "dd/MM/yyyy"),
        F.to_date(col, "MM/dd/yyyy"),
    )

def _filter_to_snapshot(df, snapshot_dt):
    return (df
            .withColumn("_snap", _parse_snapshot(F.col("snapshot_date")))
            .filter(F.col("_snap") == F.lit(snapshot_dt.date()))
            .drop("_snap"))

def process_features_bronze(snapshot_date_str, bronze_features_directory, spark):
    """
    Ingest the three monthly feature CSVs and write bronze outputs for the given snapshot date.
    Files expected under ./data:
      - feature_clickstream.csv
      - features_attributes.csv  (or feature_attributes.csv)
      - features_financials.csv  (or feature_financials.csv)
    """
    snapshot_dt = datetime.strptime(snapshot_date_str, "%Y-%m-%d")
    raw_dir = "data"

    # Clickstream snapshot (wide per-customer features)
    click_path = os.path.join(raw_dir, "feature_clickstream.csv")
    click = _read_csv(spark, click_path)
    click = (_filter_to_snapshot(click, snapshot_dt)
             .withColumn("_ingested_at_utc", F.current_timestamp()))
    (click.coalesce(1).write.mode("overwrite").option("header", True)
         .csv(f"{bronze_features_directory}bronze_feature_clickstream_{snapshot_date_str.replace('-','_')}.csv"))

    # Attributes snapshot
    attrs_path = (os.path.join(raw_dir, "features_attributes.csv")
                  if os.path.exists(os.path.join(raw_dir, "features_attributes.csv"))
                  else os.path.join(raw_dir, "feature_attributes.csv"))
    attrs = _read_csv(spark, attrs_path)
    attrs = (_filter_to_snapshot(attrs, snapshot_dt)
             .withColumn("_ingested_at_utc", F.current_timestamp()))
    (attrs.coalesce(1).write.mode("overwrite").option("header", True)
          .csv(f"{bronze_features_directory}bronze_feature_attributes_{snapshot_date_str.replace('-','_')}.csv"))

    # Financials snapshot
    fins_path = (os.path.join(raw_dir, "features_financials.csv")
                 if os.path.exists(os.path.join(raw_dir, "features_financials.csv"))
                 else os.path.join(raw_dir, "feature_financials.csv"))
    fins = _read_csv(spark, fins_path)
    fins = (_filter_to_snapshot(fins, snapshot_dt)
            .withColumn("_ingested_at_utc", F.current_timestamp()))
    (fins.coalesce(1).write.mode("overwrite").option("header", True)
         .csv(f"{bronze_features_directory}bronze_feature_financials_{snapshot_date_str.replace('-','_')}.csv"))

    print(f"✓ bronze features saved for {snapshot_date_str} -> {bronze_features_directory}")
    return {"click": click, "attr": attrs, "fin": fins}
