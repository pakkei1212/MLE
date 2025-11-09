import argparse
import os
import pyspark
from pyspark.sql.functions import col

import utils.data_processing_bronze_table  # expects process_bronze_financials()

# Usage: python bronze_financials.py --snapshotdate "2023-01-01"
def main(snapshotdate):
    print('\n\n---starting financials job---\n\n')

    spark = pyspark.sql.SparkSession.builder \
        .appName("dev") \
        .master("local[*]") \
        .getOrCreate()
    spark.sparkContext.setLogLevel("ERROR")

    date_str = snapshotdate

    bronze_financials_directory = os.path.join("datamart", "bronze", "users", "financials") + "/"
    os.makedirs(bronze_financials_directory, exist_ok=True)

    utils.data_processing_bronze_table.process_bronze_users_financials(
        date_str, bronze_financials_directory, spark
        # , financials_csv="data/features_financials.csv"  # optional override
    )

    spark.stop()
    print('\n\n---completed financials job---\n\n')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="run financials bronze job")
    parser.add_argument("--snapshotdate", type=str, required=True, help="YYYY-MM-DD")
    args = parser.parse_args()
    main(args.snapshotdate)
