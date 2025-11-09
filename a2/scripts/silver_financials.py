import argparse
import os
import pyspark

import utils.data_processing_silver_table as silver

# Usage: python silver_financials.py --snapshotdate "2023-01-01"
def main(snapshotdate: str):
    print("\n\n---starting SILVER financials job---\n")

    # Spark
    spark = pyspark.sql.SparkSession.builder.appName("dev").master("local[*]").getOrCreate()
    spark.sparkContext.setLogLevel("ERROR")

    date_str = snapshotdate

    # IO dirs
    bronze_financials_directory = os.path.join("datamart", "bronze", "users", "financials") + "/"
    silver_financials_directory = os.path.join("datamart", "silver", "users", "financials") + "/"

    os.makedirs(silver_financials_directory, exist_ok=True)

    # Run silver
    silver.process_silver_users_financials(
        date_str,
        bronze_financials_directory,
        silver_financials_directory,
        spark,
    )

    spark.stop()
    print("\n---completed SILVER financials job---\n\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Silver Financials")
    parser.add_argument("--snapshotdate", type=str, required=True, help="YYYY-MM-DD")
    args = parser.parse_args()
    main(args.snapshotdate)
