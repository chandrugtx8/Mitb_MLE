# utils/gold_builders.py
# Gold builders:
#  - consolidate_label_store: union all per-month gold_label_store_* into one dataset
#  - build_gold_training_dataset: join consolidated labels to Features-Attributes Silver
#  - write_gold_feature_store: copy Silver Features-Attributes into Gold feature_store

import os
import glob
from pyspark.sql import functions as F


def _collect_parquet_parts(dir_glob_pattern: str):
    """Return a list of part-*.parquet file paths under directories matching the glob."""
    part_files = []
    for d in glob.glob(dir_glob_pattern):
        parts = glob.glob(os.path.join(d, "part-*.parquet"))
        part_files.extend(parts)
    return part_files


def consolidate_label_store(spark, label_store_root: str, output_dir: str):
    """
    Union all monthly gold label_store_* directories into a single dataset.
      label_store_root = "datamart/gold/label_store"
      output_dir        = "datamart/gold/label_store_all.parquet"
    """
    pattern = os.path.join(label_store_root, "gold_label_store_*.parquet")
    parts = _collect_parquet_parts(pattern)
    if not parts:
        raise RuntimeError(f"No parquet parts found under {pattern}")

    df = spark.read.parquet(*parts)

    cols = ["loan_id", "Customer_ID", "label", "label_def", "snapshot_date"]
    existing = [c for c in cols if c in df.columns]
    df2 = df.select(*existing)

    df2.coalesce(1).write.mode("overwrite").parquet(output_dir)
    print(f"✓ consolidated label_store -> {output_dir} (rows: {df2.count()})")


def build_gold_training_dataset(spark, features_attributes_silver_dir: str, label_store_root: str, output_dir: str):
    """
    Join Features-Attributes Silver (all months) with Label Store (all months) into a
    single Gold training dataset, keying on loan_id = Customer_ID + '_' + snapshot_date(yyyy-MM-dd).
    """
    df_feat = spark.read.parquet(features_attributes_silver_dir)

    loan_id_feat = F.concat(
        F.col("Customer_ID"),
        F.lit("_"),
        F.date_format(F.col("snapshot_date"), "yyyy-MM-dd")
    )
    df_feat = df_feat.withColumn("loan_id", loan_id_feat)

    # Read all monthly label_store parts safely
    pattern = os.path.join(label_store_root, "gold_label_store_*.parquet")
    label_parts = _collect_parquet_parts(pattern)
    if not label_parts:
        raise RuntimeError(f"No parquet parts found under {pattern}")

    # Rename columns to avoid ambiguity on join
    df_labels = (
        spark.read.parquet(*label_parts)
             .select("loan_id", "Customer_ID", "label", "label_def", "snapshot_date")
             .withColumnRenamed("Customer_ID", "Customer_ID_label")
             .withColumnRenamed("snapshot_date", "label_snapshot_date")
    )

    joined = df_feat.join(df_labels, on="loan_id", how="inner")

    front = [
        "loan_id",
        "Customer_ID", "Name", "Age", "SSN", "Occupation",
        "snapshot_date",
        "label", "label_def", "label_snapshot_date",
        "feat_balance", "feat_utilization_bucket", "feat_trades_open",
    ]
    cols = [c for c in front if c in joined.columns] + [c for c in joined.columns if c not in front]
    joined = joined.select(*cols)

    out_path = os.path.join(output_dir, "gold_training.parquet")
    joined.coalesce(1).write.mode("overwrite").parquet(out_path)
    print(f"✓ gold training set -> {out_path} (rows: {joined.count()})")


def write_gold_feature_store(spark, features_attributes_silver_dir: str, output_dir: str):
    """
    Copy the consolidated Features-Attributes Silver into Gold:
      - datamart/gold/feature_store/feature_store_all.parquet
      - datamart/gold/feature_store/feature_store_YYYY-MM-01.parquet
    """
    df = spark.read.parquet(features_attributes_silver_dir)

    out_all = os.path.join(output_dir, "feature_store_all.parquet")
    df.coalesce(1).write.mode("overwrite").parquet(out_all)

    months = [r["m"] for r in df.select(
        F.date_format("snapshot_date", "yyyy-MM-01").alias("m")
    ).distinct().collect()]

    for m in months:
        df_m = df.filter(F.date_format("snapshot_date", "yyyy-MM-01") == F.lit(m))
        out_m = os.path.join(output_dir, f"feature_store_{m}.parquet")
        df_m.coalesce(1).write.mode("overwrite").parquet(out_m)

    print(f"✓ gold feature store -> {output_dir} (months: {len(months)}, rows: {df.count()})")
