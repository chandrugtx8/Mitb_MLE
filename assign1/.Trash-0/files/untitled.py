#
import os
import glob
from datetime import datetime
import pyspark
from pyspark.sql import functions as F

# --- your Lab 2 utils ---
import utils.data_processing_bronze_table as lms_bronze
import utils.data_processing_silver_table as lms_silver
import utils.data_processing_gold_table as lms_gold

# --- new: features bronze ---
from utils.features_bronze_table import process_features_bronze

# -------------------------
# Spark session
# -------------------------
spark = pyspark.sql.SparkSession.builder \
    .appName("cs611-a1") \
    .master("local[*]") \
    .getOrCreate()
spark.sparkContext.setLogLevel("ERROR")

# -------------------------
# Config (monthly backfill)
# -------------------------
start_date_str = "2023-01-01"
end_date_str   = "2024-12-01"

def generate_first_of_month_dates(start_date_str, end_date_str):
    start_date = datetime.strptime(start_date_str, "%Y-%m-%d")
    end_date   = datetime.strptime(end_date_str,   "%Y-%m-%d")
    firsts = []
    cur = datetime(start_date.year, start_date.month, 1)
    while cur <= end_date:
        firsts.append(cur.strftime("%Y-%m-%d"))
        if cur.month == 12:
            cur = datetime(cur.year + 1, 1, 1)
        else:
            cur = datetime(cur.year, cur.month + 1, 1)
    return firsts

dates_str_lst = generate_first_of_month_dates(start_date_str, end_date_str)
print("Months to process:", dates_str_lst)

# -------------------------
# Output dirs
# -------------------------
bronze_lms_directory     = "datamart/bronze/lms/"
silver_loan_daily_dir    = "datamart/silver/loan_daily/"
gold_label_store_dir     = "datamart/gold/label_store/"
bronze_features_directory = "datamart/bronze/features/"

for d in [bronze_lms_directory, silver_loan_daily_dir, gold_label_store_dir, bronze_features_directory]:
    os.makedirs(d, exist_ok=True)

# -------------------------
# 1) LMS pipeline (Lab 2)
# -------------------------
for date_str in dates_str_lst:
    lms_bronze.process_bronze_table(date_str, bronze_lms_directory, spark)

for date_str in dates_str_lst:
    lms_silver.process_silver_table(date_str, bronze_lms_directory, silver_loan_daily_dir, spark)

for date_str in dates_str_lst:
    lms_gold.process_labels_gold_table(date_str, silver_loan_daily_dir, gold_label_store_dir, spark, dpd=30, mob=6)

# simple sanity check
files_list = [gold_label_store_dir + os.path.basename(f) for f in glob.glob(os.path.join(gold_label_store_dir, '*'))]
df_labels = spark.read.option("header", "true").parquet(*files_list)
print("label_store row_count:", df_labels.count())
df_labels.show(5, truncate=False)

# -------------------------
# 2) Features pipeline (bronze)
# -------------------------
for date_str in dates_str_lst:
    process_features_bronze(date_str, bronze_features_directory, spark)

print("features bronze written to:", bronze_features_directory)
print("All done.")
