import argparse
import os
import pyspark

import utils.data_processing_silver_table as silver

# Usage: python silver_label_store.py --snapshotdate "2023-01-01"
def main(snapshotdate: str):
    print("\n\n---starting SILVER label/loans job---\n")

    # Spark
    spark = pyspark.sql.SparkSession.builder.appName("dev").master("local[*]").getOrCreate()
    spark.sparkContext.setLogLevel("ERROR")

    date_str = snapshotdate

    # IO dirs
    bronze_lms_directory = os.path.join("datamart", "bronze", "lms") + "/"
    silver_loan_daily_directory = os.path.join("datamart", "silver", "lms") + "/"

    os.makedirs(silver_loan_daily_directory, exist_ok=True)

    # Run silver loans (label store)
    silver.process_silver_loans(
        date_str,
        bronze_lms_directory,
        silver_loan_daily_directory,
        spark,
    )

    spark.stop()
    print("\n---completed SILVER label/loans job---\n\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Silver Label/Loans")
    parser.add_argument("--snapshotdate", type=str, required=True, help="YYYY-MM-DD")
    args = parser.parse_args()
    main(args.snapshotdate)
