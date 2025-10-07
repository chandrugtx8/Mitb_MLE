
# utils/features_attributes_silver_table.py
# Builds a consolidated Silver table from Features Bronze CSVs
# Fixes the Occupation cleaning to avoid datatype and unresolved regex errors.

import os
from pyspark.sql import functions as F, types as T


def process_features_attributes_silver(spark, input_glob: str, output_dir: str):
    """
    Read all Feature Bronze CSVs (input_glob), clean fields, and write a single
    Silver Features-Attributes Parquet dataset into output_dir (coalesce(1)).

    Parameters
    ----------
    spark : SparkSession
    input_glob : str
        Glob path to all features bronze CSV files, e.g. datamart/bronze/features/bronze_features_*.csv
    output_dir : str
        Output directory for a single parquet dataset (e.g., datamart/silver/features_attributes).
    """

    # Read CSVs with an explicit schema to avoid inference edge cases
    schema = T.StructType([
        T.StructField("Customer_ID", T.StringType(), True),
        T.StructField("Name",        T.StringType(), True),
        T.StructField("Age",         T.StringType(), True),   # cast later
        T.StructField("SSN",         T.StringType(), True),
        T.StructField("Occupation",  T.StringType(), True),
        T.StructField("snapshot_date",    T.StringType(), True),
        T.StructField("_ingested_at_utc", T.StringType(), True),
        T.StructField("feat_balance",              T.IntegerType(), True),
        T.StructField("feat_utilization_bucket",   T.IntegerType(), True),
        T.StructField("feat_trades_open",          T.IntegerType(), True),
    ])

    df_in = spark.read.option("header", True).schema(schema).csv(input_glob)

    # ---------------------
    # Cleaning (boolean-safe)
    # ---------------------

    # Customer_ID
    df1 = df_in.withColumn("Customer_ID", F.trim(F.col("Customer_ID")))

    # Name cleaning
    name_raw = F.regexp_replace(F.col("Name"), r'["\'`]', "")
    name_trim = F.trim(F.regexp_replace(name_raw, r"\s+", " "))
    name_norm = F.initcap(F.lower(name_trim))
    name_clean = F.when(F.length(name_norm) == 0, F.lit("Unknown")).otherwise(name_norm)
    df1 = df1.withColumn("Name", name_clean)

    # Age to int; negatives -> null
    age_int = F.col("Age").cast("int")
    df1 = df1.withColumn("Age", F.when(age_int < 0, F.lit(None).cast("int")).otherwise(age_int))

    # SSN digits + hyphen
    df1 = df1.withColumn("SSN", F.regexp_replace(F.col("SSN").cast("string"), r"[^0-9\-]", ""))

    # Occupation: treat null/empty or no letters as 'Unknown'
    occ_trim = F.trim(F.col("Occupation"))
    occ_is_missing = occ_trim.isNull() | (F.length(occ_trim) == 0)
    occ_letters = F.regexp_replace(occ_trim, r"[^A-Za-z]+", "")
    occ_has_no_letters = F.length(occ_letters) == 0
    occ_base = F.initcap(F.lower(occ_trim))
    occ_final = F.when(occ_is_missing | occ_has_no_letters, F.lit("Unknown")).otherwise(occ_base)
    df1 = df1.withColumn("Occupation", occ_final)

    # Dates
    df1 = df1.withColumn("snapshot_date", F.to_date("snapshot_date"))
    df1 = df1.withColumn("_ingested_at_utc", F.to_timestamp("_ingested_at_utc"))

    # Keep columns in a consistent order
    cols = [
        "Customer_ID", "Name", "Age", "SSN", "Occupation",
        "snapshot_date", "_ingested_at_utc",
        "feat_balance", "feat_utilization_bucket", "feat_trades_open",
    ]
    df_out = df1.select(*cols)

    # Write one consolidated parquet dataset
    df_out.coalesce(1).write.mode("overwrite").parquet(output_dir)

    # Small confirmation print
    print(f"features-attributes silver written to: {output_dir}")
    print("row_count:", df_out.count())
    df_out.show(5, truncate=False)