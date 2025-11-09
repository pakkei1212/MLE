import argparse
import os
import pyspark

import utils.data_processing_silver_table as silver

# Usage: python silver_attributes.py --snapshotdate "2023-01-01"
def main(snapshotdate: str):
    print("\n\n---starting SILVER attributes job---\n")

    # Spark
    spark = pyspark.sql.SparkSession.builder.appName("dev").master("local[*]").getOrCreate()
    spark.sparkContext.setLogLevel("ERROR")

    date_str = snapshotdate

    # IO dirs
    bronze_attributes_directory = os.path.join("datamart", "bronze", "users", "attributes") + "/"
    silver_attributes_directory = os.path.join("datamart", "silver", "users", "attributes") + "/"

    os.makedirs(silver_attributes_directory, exist_ok=True)

    # Run silver
    silver.process_silver_users_attributes(
        date_str,
        bronze_attributes_directory,
        silver_attributes_directory,
        spark,
    )

    spark.stop()
    print("\n---completed SILVER attributes job---\n\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Silver Attributes")
    parser.add_argument("--snapshotdate", type=str, required=True, help="YYYY-MM-DD")
    args = parser.parse_args()
    main(args.snapshotdate)
