import argparse
import os
import pyspark

import utils.data_processing_silver_table as silver

# Usage: python silver_clickstream.py --snapshotdate "2023-01-01"
def main(snapshotdate: str):
    print("\n\n---starting SILVER clickstream job---\n")

    # Spark
    spark = pyspark.sql.SparkSession.builder.appName("dev").master("local[*]").getOrCreate()
    spark.sparkContext.setLogLevel("ERROR")

    date_str = snapshotdate

    # IO dirs
    bronze_clickstream_directory = os.path.join("datamart", "bronze", "users", "clickstream") + "/"
    silver_clickstream_directory = os.path.join("datamart", "silver", "users", "clickstream") + "/"

    os.makedirs(silver_clickstream_directory, exist_ok=True)

    # Run silver
    silver.process_silver_users_clickstream(
        date_str,
        bronze_clickstream_directory,
        silver_clickstream_directory,
        spark,
    )

    spark.stop()
    print("\n---completed SILVER clickstream job---\n\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Silver Clickstream")
    parser.add_argument("--snapshotdate", type=str, required=True, help="YYYY-MM-DD")
    args = parser.parse_args()
    main(args.snapshotdate)
