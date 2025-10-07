# utils/features_attributes_silver_table.py
# Build a single Silver Features-Attributes parquet dataset from all Features Bronze CSVs.

from pyspark.sql import functions as F, types as T


def process_features_attributes_silver(spark, input_glob: str, output_dir: str):
    # Read all bronze feature CSVs
    df = (
        spark.read
        .option("header", True)
        .csv(input_glob)
    )

    # Normalize/clean columns
    # Trim basic string fields
    df = (
        df.withColumn("Customer_ID", F.trim(F.col("Customer_ID")))
          .withColumn("Name", F.trim(F.col("Name")))
          .withColumn("SSN", F.trim(F.col("SSN")))
          .withColumn("Occupation", F.trim(F.col("Occupation")))
    )

    # Name cleaning
    name_raw = F.regexp_replace(F.col("Name"), r'["\'`]', "")
    name_trim = F.trim(F.regexp_replace(name_raw, r"\s+", " "))
    name_norm = F.initcap(F.lower(name_trim))
    name_clean = F.when(F.length(name_norm) == 0, F.lit("Unknown")).otherwise(name_norm)
    df = df.withColumn("Name", name_clean)

    # Age
    age_int = F.col("Age").cast("int")
    df = df.withColumn("Age", F.when(age_int < 0, F.lit(None).cast("int")).otherwise(age_int))

    # SSN
    df = df.withColumn("SSN", F.regexp_replace(F.col("SSN").cast("string"), r"[^0-9\-]", ""))

    # Occupation (boolean-safe)
    occ_trim = F.trim(F.col("Occupation"))
    occ_is_missing = occ_trim.isNull() | (F.length(occ_trim) == 0)
    occ_letters = F.regexp_replace(occ_trim, r"[^A-Za-z]+", "")
    occ_has_no_letters = F.length(occ_letters) == 0
    occ_base = F.initcap(F.lower(occ_trim))
    occ_final = F.when(occ_is_missing | occ_has_no_letters, F.lit("Unknown")).otherwise(occ_base)
    df = df.withColumn("Occupation", occ_final)

    # Dates
    df = df.withColumn("snapshot_date", F.to_date("snapshot_date"))
    df = df.withColumn("_ingested_at_utc", F.to_timestamp("_ingested_at_utc"))

    # Cast feature columns (they arrive as strings from CSV)
    df = (
        df.withColumn("feat_balance", F.col("feat_balance").cast("int"))
          .withColumn("feat_utilization_bucket", F.col("feat_utilization_bucket").cast("int"))
          .withColumn("feat_trades_open", F.col("feat_trades_open").cast("int"))
    )

    # Reorder columns
    ordered = [
        "Customer_ID", "Name", "Age", "SSN", "Occupation",
        "snapshot_date", "_ingested_at_utc",
        "feat_balance", "feat_utilization_bucket", "feat_trades_open"
    ]
    existing = [c for c in ordered if c in df.columns]
    df = df.select(*existing)

    # Write
    df.coalesce(1).write.mode("overwrite").parquet(output_dir)

    # Log a small sample
    cnt = df.count()
    print(f"features-attributes silver written to: {output_dir}")
    print("row_count:", cnt)
    df.show(5, truncate=False)
