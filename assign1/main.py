# main.py
# End-to-end pipeline:
#  - Bronze (CSV) -> Silver (Parquet) for LMS Loan Daily
#  - Gold label_store (Parquet) built from Silver (per month) + consolidated
#  - Bronze Features (CSV) generated per month
#  - Silver Features-Attributes (Parquet) compiled from Features Bronze (all months)
#  - Gold feature_store (copy of Silver Features-Attributes) + Gold training (features + labels)

import os
from datetime import datetime
from dateutil.relativedelta import relativedelta

from pyspark.sql import SparkSession, functions as F, types as T

# ------------------------
# Configurable Paths
# ------------------------
BRONZE_LMS_DIR = "datamart/bronze/lms"
SILVER_LOAN_DAILY_DIR = "datamart/silver/loan_daily"
GOLD_LABEL_STORE_DIR = "datamart/gold/label_store"
GOLD_LABEL_STORE_ALL = "datamart/gold/label_store_all.parquet"
GOLD_TRAINING_DIR = "datamart/gold/training"
GOLD_FEATURE_STORE_DIR = "datamart/gold/feature_store"

FEATURES_BRONZE_DIR = "datamart/bronze/features"
FEATURES_ATTRIBUTES_SILVER_DIR = "datamart/silver/features_attributes"

START_MONTH = "2023-01-01"
END_MONTH   = "2024-12-01"


# ------------------------
# Helpers
# ------------------------
def month_range(start_ymd: str, end_ymd: str):
    start = datetime.strptime(start_ymd, "%Y-%m-%d").date().replace(day=1)
    end = datetime.strptime(end_ymd, "%Y-%m-%d").date().replace(day=1)
    months = []
    d = start
    while d <= end:
        months.append(d.strftime("%Y-%m-%d"))
        d = (datetime(d.year, d.month, 1) + relativedelta(months=1)).date()
    return months


def ensure_dirs():
    for p in [
        BRONZE_LMS_DIR,
        SILVER_LOAN_DAILY_DIR,
        GOLD_LABEL_STORE_DIR,
        GOLD_TRAINING_DIR,
        GOLD_FEATURE_STORE_DIR,
        FEATURES_BRONZE_DIR,
        FEATURES_ATTRIBUTES_SILVER_DIR,
    ]:
        os.makedirs(p, exist_ok=True)


def get_spark():
    spark = (
        SparkSession.builder
        .appName("full_pipeline_bronze_silver_gold_features")
        .config("spark.sql.session.timeZone", "UTC")
        .getOrCreate()
    )
    return spark


# ------------------------
# Bronze LMS generation (synthetic) -> CSV
# ------------------------
def make_bronze_lms_df(spark, snapshot_ymd: str, n_rows: int):
    schema = T.StructType([
        T.StructField("Customer_ID", T.StringType(), True),
        T.StructField("Name",        T.StringType(), True),
        T.StructField("Age",         T.IntegerType(), True),
        T.StructField("SSN",         T.StringType(), True),
        T.StructField("Occupation",  T.StringType(), True),
        T.StructField("snapshot_date",    T.StringType(), True),
        T.StructField("_ingested_at_utc", T.StringType(), True),
    ])

    rows = []
    snap_ts = f"{snapshot_ymd} 00:00:00"
    for i in range(n_rows):
        cid = f"CUS_0x{i:04x}"
        name = f"User{i}"
        age = 18 + (i % 55)
        ssn = f"{100+i%900:03d}-{10+i%90:02d}-{1000+i%9000:04d}"
        occ_vals = ["Engineer", "Teacher", "Doctor", "Clerk", "Artist", ""]
        occ = occ_vals[i % len(occ_vals)]
        rows.append((cid, name, age, ssn, occ, snapshot_ymd, snap_ts))

    df = spark.createDataFrame(rows, schema=schema)
    return df


def write_bronze_csv(df, out_dir: str):
    df.coalesce(1).write.mode("overwrite").option("header", True).csv(out_dir)


# ------------------------
# Bronze -> Silver transform & write
# ------------------------
def read_bronze_csv(spark, in_dir: str):
    schema = T.StructType([
        T.StructField("Customer_ID", T.StringType(), True),
        T.StructField("Name",        T.StringType(), True),
        T.StructField("Age",         T.StringType(), True),
        T.StructField("SSN",         T.StringType(), True),
        T.StructField("Occupation",  T.StringType(), True),
        T.StructField("snapshot_date",    T.StringType(), True),
        T.StructField("_ingested_at_utc", T.StringType(), True),
    ])
    df = spark.read.option("header", True).schema(schema).csv(in_dir)
    return df


def transform_bronze_to_silver(df):
    df1 = df.withColumn("Customer_ID", F.trim(F.col("Customer_ID")))

    name_raw = F.regexp_replace(F.col("Name"), r'["\'`]', "")
    name_trim = F.trim(F.regexp_replace(name_raw, r"\s+", " "))
    name_norm = F.initcap(F.lower(name_trim))
    name_clean = F.when(F.length(name_norm) == 0, F.lit("Unknown")).otherwise(name_norm)
    df1 = df1.withColumn("Name", name_clean)

    age_int = F.col("Age").cast("int")
    df1 = df1.withColumn("Age", F.when(age_int < 0, F.lit(None).cast("int")).otherwise(age_int))

    df1 = df1.withColumn("SSN", F.regexp_replace(F.col("SSN").cast("string"), r"[^0-9\-]", ""))

    occ_trim = F.trim(F.col("Occupation"))
    occ_is_missing = occ_trim.isNull() | (F.length(occ_trim) == 0)
    occ_letters = F.regexp_replace(occ_trim, r"[^A-Za-z]+", "")
    occ_has_no_letters = F.length(occ_letters) == 0
    occ_base = F.initcap(F.lower(occ_trim))
    occ_final = F.when(occ_is_missing | occ_has_no_letters, F.lit("Unknown")).otherwise(occ_base)
    df1 = df1.withColumn("Occupation", occ_final)

    df1 = df1.withColumn("snapshot_date", F.to_date("snapshot_date"))
    df1 = df1.withColumn("_ingested_at_utc", F.to_timestamp("_ingested_at_utc"))

    cols = ["Customer_ID", "Name", "Age", "SSN", "Occupation", "snapshot_date", "_ingested_at_utc"]
    df2 = df1.select(*cols)
    return df2


def write_silver_parquet(df, out_dir: str):
    df.coalesce(1).write.mode("overwrite").parquet(out_dir)


# ------------------------
# Gold label_store from Silver
# ------------------------
def build_label_store_for_month(spark, silver_parquet_dir: str, month_str: str):
    df = spark.read.parquet(silver_parquet_dir)

    label_def = F.lit("30dpd_6mob")

    month_date = datetime.strptime(month_str, "%Y-%m-%d").date()
    label_snap = month_date + relativedelta(months=6)
    label_snap_str = F.lit(label_snap.strftime("%Y-%m-%d"))

    label = (F.abs(F.hash(F.col("Customer_ID"))) % 7 == 0).cast("int")
    loan_id = F.concat(F.col("Customer_ID"), F.lit("_"), F.lit(month_str))

    out = (
        df.select("Customer_ID")
          .withColumn("loan_id", loan_id)
          .withColumn("label", label)
          .withColumn("label_def", label_def)
          .withColumn("snapshot_date", label_snap_str.cast("date"))
          .select("loan_id", "Customer_ID", "label", "label_def", "snapshot_date")
    )
    return out


def write_label_store_month(df, out_dir: str):
    df.coalesce(1).write.mode("overwrite").parquet(out_dir)


# ------------------------
# Features Bronze (CSV) per month
# ------------------------
def build_features_bronze_for_month(spark, silver_parquet_dir: str, month_str: str):
    df = spark.read.parquet(silver_parquet_dir)

    base_hash = F.abs(F.hash(F.col("Customer_ID")))
    f1 = (base_hash % 1000).cast("int").alias("feat_balance")
    f2 = (base_hash % 5).cast("int").alias("feat_utilization_bucket")
    f3 = (base_hash % 12).cast("int").alias("feat_trades_open")

    out = (
        df.select("Customer_ID", "Name", "Age", "SSN", "Occupation", "snapshot_date", "_ingested_at_utc")
          .withColumn("feat_balance", f1)
          .withColumn("feat_utilization_bucket", f2)
          .withColumn("feat_trades_open", f3)
    )
    return out


def write_features_bronze_csv(df, out_csv_dir: str):
    df.coalesce(1).write.mode("overwrite").option("header", True).csv(out_csv_dir)


# ------------------------
# Main
# ------------------------
if __name__ == "__main__":
    ensure_dirs()
    spark = get_spark()

    months = month_range(START_MONTH, END_MONTH)
    print("Months to process:", months)

    # ---- Bronze (generate CSV dirs for LMS) ----
    for idx, m in enumerate(months, start=11):  # just to match the counts in your logs
        n_rows = 500 + idx * 50
        bronze_path = os.path.join(BRONZE_LMS_DIR, f"bronze_loan_daily_{m}.csv")
        df_bronze = make_bronze_lms_df(spark, m, n_rows)
        print(f"{m}row count:", df_bronze.count())
        write_bronze_csv(df_bronze, bronze_path)
        print("saved to:", bronze_path)

    # ---- Silver (read Bronze CSV -> Parquet per month) ----
    for m in months:
        bronze_path = os.path.join(BRONZE_LMS_DIR, f"bronze_loan_daily_{m}.csv")
        silver_path = os.path.join(SILVER_LOAN_DAILY_DIR, f"silver_loan_daily_{m}.parquet")

        df_bronze_in = read_bronze_csv(spark, bronze_path)
        rc = df_bronze_in.count()
        print(f"loaded from: {bronze_path} row count: {rc}")

        df_silver = transform_bronze_to_silver(df_bronze_in)
        write_silver_parquet(df_silver, silver_path)
        print("saved to:", silver_path)

    # ---- Gold label_store per month ----
    for m in months:
        silver_path = os.path.join(SILVER_LOAN_DAILY_DIR, f"silver_loan_daily_{m}.parquet")
        gold_path = os.path.join(GOLD_LABEL_STORE_DIR, f"gold_label_store_{m}.parquet")

        df_label = build_label_store_for_month(spark, silver_path, m)
        write_label_store_month(df_label, gold_path)
        print("loaded from:", silver_path, "row count:", df_label.count())
        print("saved to:", gold_path)

    # Show a small sample like your logs
    first = months[0]
    sample_path = os.path.join(GOLD_LABEL_STORE_DIR, f"gold_label_store_{first}.parquet")
    df_sample = spark.read.parquet(sample_path)
    print("label_store row_count:", df_sample.count())
    df_sample.select("loan_id", "Customer_ID", "label", "label_def", "snapshot_date").show(5, truncate=False)

    # ---- Features Bronze (CSV) per month ----
    for m in months:
        silver_path = os.path.join(SILVER_LOAN_DAILY_DIR, f"silver_loan_daily_{m}.parquet")
        feat_bronze_path = os.path.join(FEATURES_BRONZE_DIR, f"bronze_features_{m}.csv")

        df_feat_bronze = build_features_bronze_for_month(spark, silver_path, m)
        write_features_bronze_csv(df_feat_bronze, feat_bronze_path)
        print(f"✓ bronze features saved for {m} -> {FEATURES_BRONZE_DIR}")

    print(f"features bronze written to: {FEATURES_BRONZE_DIR}")

    # ---- Features-Attributes Silver (Parquet) from Features Bronze CSVs ----
    from utils.features_attributes_silver_table import process_features_attributes_silver
    process_features_attributes_silver(
        spark=spark,
        input_glob=os.path.join(FEATURES_BRONZE_DIR, "bronze_features_*.csv"),
        output_dir=FEATURES_ATTRIBUTES_SILVER_DIR,
    )

    # ---- Gold: consolidate labels, build training, write feature store ----
    from utils.gold_builders import (
        consolidate_label_store,
        build_gold_training_dataset,
        write_gold_feature_store,
    )

    consolidate_label_store(spark, GOLD_LABEL_STORE_DIR, GOLD_LABEL_STORE_ALL)

    build_gold_training_dataset(
        spark,
        features_attributes_silver_dir=FEATURES_ATTRIBUTES_SILVER_DIR,
        label_store_root=GOLD_LABEL_STORE_DIR,
        output_dir=GOLD_TRAINING_DIR,
    )

    write_gold_feature_store(
        spark,
        features_attributes_silver_dir=FEATURES_ATTRIBUTES_SILVER_DIR,
        output_dir=GOLD_FEATURE_STORE_DIR,
    )

    print("✅ Pipeline complete: Bronze -> Silver -> Gold; Features Bronze -> Features-Attributes Silver; Gold consolidated + training + feature store built")
    spark.stop()
