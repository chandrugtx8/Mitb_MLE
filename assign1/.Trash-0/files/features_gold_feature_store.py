# utils/features_gold_feature_store.py
import os
from pyspark.sql import functions as F

def _read_silver_csv(spark, path):
    # Each “*.csv” path here is actually a folder containing a single part-*.csv
    return (
        spark.read
             .option("header", True)
             .option("inferSchema", True)   # restore numeric types
             .csv(path)
    )

def process_feature_store_gold(
    snapshot_date_str: str,
    silver_features_attr_dir: str,
    silver_features_fin_dir: str,
    gold_feature_store_dir: str,
    spark,
    label_store_dir: str | None = None,
    write_csv_copy: bool = True,
):
    """
    Build one month of the feature store by joining the two silver feature tables.
    Optionally attaches labels (left-join).
    Writes Parquet (and also a single-part CSV copy if write_csv_copy=True).
    """
    date_tag = snapshot_date_str.replace("-", "_")

    # -------- read silver inputs --------
    attr_path = os.path.join(
        silver_features_attr_dir, f"silver_feature_attributes_{date_tag}.csv"
    )
    fin_path = os.path.join(
        silver_features_fin_dir, f"silver_feature_financials_{date_tag}.csv"
    )

    attr = _read_silver_csv(spark, attr_path)
    fin  = _read_silver_csv(spark, fin_path)

    # normalize keys
    if "snapshot_date" in attr.columns:
        attr = attr.withColumn("snapshot_date", F.to_date("snapshot_date"))
    if "snapshot_date" in fin.columns:
        fin  = fin.withColumn("snapshot_date", F.to_date("snapshot_date"))

    # keep only one copy of the join keys if they exist on both sides
    join_keys = ["Customer_ID", "snapshot_date"]
    for k in join_keys:
        if k in attr.columns:
            attr = attr.withColumn(k, F.trim(F.col(k)))
        if k in fin.columns:
            fin  = fin.withColumn(k, F.trim(F.col(k)))

    # -------- inner join: only rows present in both tables --------
    features = attr.join(fin, join_keys, how="inner")

    # -------- optionally attach label --------
    if label_store_dir is not None:
        label_path = os.path.join(
            label_store_dir, f"gold_label_store_{date_tag}.parquet"
        )
        try:
            labels = spark.read.parquet(label_path).select(
                "Customer_ID", "snapshot_date", "label", "label_def"
            )
            features = features.join(labels, join_keys, how="left")
        except Exception as e:
            print(f"[warn] could not read labels for {snapshot_date_str}: {e}")

    # -------- hygiene: dedupe + fill nulls --------
    features = features.dropDuplicates(join_keys)

    # numeric → 0, categoricals → 'Unknown'
    num_prefixes = ("int", "bigint", "double", "float", "decimal", "smallint", "tinyint")
    numeric_cols = [c for c, t in features.dtypes if t.startswith(num_prefixes)]
    cat_cols     = [c for c in features.columns if c not in numeric_cols]

    for c in numeric_cols:
        features = features.withColumn(c, F.when(F.col(c).isNull(), F.lit(0)).otherwise(F.col(c)))
    for c in cat_cols:
        if c not in join_keys:  # don't touch the keys
            features = features.withColumn(c, F.when(F.col(c).isNull(), F.lit("Unknown")).otherwise(F.col(c)))

    # -------- write outputs --------
    os.makedirs(gold_feature_store_dir, exist_ok=True)

    out_parquet = os.path.join(gold_feature_store_dir, f"gold_feature_store_{date_tag}.parquet")
    features.write.mode("overwrite").parquet(out_parquet)

    if write_csv_copy:
        out_csv = os.path.join(gold_feature_store_dir, f"gold_feature_store_{date_tag}.csv")
        (features.coalesce(1)
                 .write.mode("overwrite")
                 .option("header", True)
                 .csv(out_csv))

    print(f"✓ feature store saved for {snapshot_date_str} -> {gold_feature_store_dir}")
    return features
