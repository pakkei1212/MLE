import argparse
import os
import pyspark
from pyspark.sql.functions import col  # keeps parity with your imports

import utils.data_processing_bronze_table  # expects process_bronze_clickstream()

# Usage: python bronze_clickstream.py --snapshotdate "2023-01-01"
def main(snapshotdate):
    print('\n\n---starting clickstream job---\n\n')

    # Initialize SparkSession
    spark = pyspark.sql.SparkSession.builder \
        .appName("dev") \
        .master("local[*]") \
        .getOrCreate()
    spark.sparkContext.setLogLevel("ERROR")

    date_str = snapshotdate

    # Bronze output directory
    bronze_clickstream_directory = os.path.join("datamart", "bronze", "users", "clickstream") + "/"
    os.makedirs(bronze_clickstream_directory, exist_ok=True)

    # Run processing
    utils.data_processing_bronze_table.process_bronze_users_clickstream(
        date_str, bronze_clickstream_directory, spark
        # , clickstream_csv="data/feature_clickstream.csv"  # optional override
    )

    spark.stop()
    print('\n\n---completed clickstream job---\n\n')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="run clickstream bronze job")
    parser.add_argument("--snapshotdate", type=str, required=True, help="YYYY-MM-DD")
    args = parser.parse_args()
    main(args.snapshotdate)
