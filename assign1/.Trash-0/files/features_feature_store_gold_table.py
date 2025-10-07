# utils/features_feature_store_gold_table.py
import os
from pyspark.sql import functions as F, types as T

def _read_silver_csv(spark, base_dir, snapshot_date_str, kind):
    """
    kind: "attributes" or "financials".
    Reads a monthly silver CSV partition folder like:
      silver_feature_{kind}_YYYY_MM_DD.csv/part-*.csv
    """
    folder = f"silver_feature_{kind}_{snapshot_date_str.replace('-', '_')}.csv"
    path = os.path.join(base_dir, folder)
    return spark.read.option("header", True).csv(path)

def process_feature_store_gold(
    snapshot_date_str: str,
    silver_features_attr_dir: str,
    silver_features_fin_dir: str,
    gold_feature_store_dir: str,
    spark,
):
    # ---- 1) Read monthly SILVER partitions ----
    df_attr = _read_silver_csv(spark, silver_features_attr_dir, snapshot_date_str, "attributes")
    df_fin  = _read_silver_csv(spark, silver_features_fin_dir,  snapshot_date_str, "financials")

    # ---- 2) Normalize keys & dates ----
    if "snapshot_date" in df_attr.columns:
        df_attr = df_attr.withColumn("snapshot_date", F.to_date("snapshot_date"))
    if "snapshot_date" in df_fin.columns:
        df_fin = df_fin.withColumn("snapshot_date", F.to_date("snapshot_date"))

    df_attr = df_attr.withColumn("Customer_ID", F.trim(F.col("Customer_ID")))
    df_fin  = df_fin.withColumn("Customer_ID",  F.trim(F.col("Customer_ID")))

    # Avoid duplicate name on join
    if "_ingested_at_utc" in df_attr.columns:
        df_attr = df_attr.withColumnRenamed("_ingested_at_utc", "_ingested_at_utc_attr")
    if "_ingested_at_utc" in df_fin.columns:
        df_fin = df_fin.withColumnRenamed("_ingested_at_utc", "_ingested_at_utc_fin")

    # ---- 3) Cast numerics (silver CSVs load as strings) ----
    if "Age" in df_attr.columns:
        df_attr = df_attr.withColumn("Age", F.col("Age").cast(T.IntegerType()))

    fin_int = [
        "Num_Bank_Accounts","Num_Credit_Card","Interest_Rate","Num_of_Loan",
        "Delay_from_due_date","Num_of_Delayed_Payment","Num_Credit_Inquiries",
        "credit_history_age_year","credit_history_age_month",
    ]
    fin_float = [
        "Annual_Income","Monthly_Inhand_Salary","Changed_Credit_Limit",
        "Outstanding_Debt","Credit_Utilization_Ratio","Total_EMI_per_month",
        "Amount_invested_monthly","Monthly_Balance","Credit_History_Months",
    ]

    # If months total missing, derive from year/month
    if ("credit_history_age_year" in df_fin.columns
            and "credit_history_age_month" in df_fin.columns
            and "Credit_History_Months" not in df_fin.columns):
        df_fin = df_fin.withColumn(
            "Credit_History_Months",
            (F.col("credit_history_age_year") * 12 + F.col("credit_history_age_month")).cast(T.IntegerType())
        )

    # OHE & loan-flag columns
    ohe_cols = [c for c in df_fin.columns if c.startswith("ohe__")]
    loan_flag_cols = [
        "mortgage_loan","auto_loan","credit_builder_loan","personal_loan",
        "not_specified","student_loan","home_equity_loan","payday_loan","debt_consolidation_loan"
    ]

    for c in fin_int + ohe_cols + loan_flag_cols:
        if c in df_fin.columns:
            df_fin = df_fin.withColumn(c, F.col(c).cast(T.IntegerType()))
    for c in fin_float:
        if c in df_fin.columns:
            df_fin = df_fin.withColumn(c, F.col(c).cast(T.DoubleType()))

    # ---- 4) Dedupe per source ----
    if {"snapshot_date","Customer_ID"}.issubset(df_attr.columns):
        df_attr = df_attr.dropDuplicates(["snapshot_date","Customer_ID"])
    if {"snapshot_date","Customer_ID"}.issubset(df_fin.columns):
        df_fin = df_fin.dropDuplicates(["snapshot_date","Customer_ID"])

    # ---- 5) Join (inner keeps rows present in both) ----
    join_keys = ["Customer_ID", "snapshot_date"]
    df = df_attr.join(df_fin, on=join_keys, how="inner")

    # ---- 6) Fill nulls: numeric -> 0, strings -> "Unknown" ----
    num_types = {"int", "bigint", "double", "float", "decimal", "smallint", "tinyint"}
    numeric_cols = [c for c, t in df.dtypes if any(t.startswith(nt) for nt in num_types)]
    string_cols  = [c for c, t in df.dtypes if t == "string"]

    for c in numeric_cols:
        df = df.withColumn(c, F.when(F.col(c).isNull(), F.lit(0)).otherwise(F.col(c)))
    for c in string_cols:
        df = df.withColumn(c, F.when(F.col(c).isNull(), F.lit("Unknown")).otherwise(F.col(c)))

    # ---- 7) Write GOLD feature store (CSV per month) ----
    out_dir = os.path.join(
        gold_feature_store_dir,
        f"feature_store_{snapshot_date_str.replace('-', '_')}.csv"
    )
    (df.coalesce(1)
       .write.mode("overwrite")
       .option("header", True)
       .csv(out_dir))

    print(f"saved feature store (gold) -> {out_dir}")
    return df